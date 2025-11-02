"""Data bridge and process coordination for the Streamlit monitoring UI.

This module exposes a thread-safe :class:`DataBridge` that keeps track of the
latest video frame and metric data (bar and line charts). A ``multiprocessing
manager`` is used to share the bridge instance across processes so that both the
existing ``main.py`` loop and the Streamlit UI can exchange information without
blocking one another.

Usage highlights
----------------
* ``start_bridge_server`` spins up a manager process that hosts the bridge.
* ``connect_to_bridge`` / ``connect_to_bridge_from_env`` return a proxy to the
  shared bridge so that producers (``main.py``) and consumers (Streamlit UI)
  can push/pull data safely.
* The bridge optionally generates synthetic demo data which is helpful when the
  game loop is not running.

The exposed interface strictly follows the requirements from the user request:

``get_video_frame`` -> ``np.ndarray``
``get_bar_chart_data`` -> ``List[dict]``
``get_line_chart_data`` -> ``List[dict]``
``update_data(data_type: str, data: Any)`` -> ``None``

All interactions are guarded by locks to ensure thread-safety.
"""
from __future__ import annotations

import math
import os
import secrets
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from multiprocessing.managers import BaseManager

# ---------------------------------------------------------------------------
# Environment keys shared between processes
# ---------------------------------------------------------------------------
ENV_HOST = "DATA_BRIDGE_HOST"
ENV_PORT = "DATA_BRIDGE_PORT"
ENV_AUTHKEY = "DATA_BRIDGE_AUTHKEY"

_DEFAULT_AUTHKEY = b"mediabridge-secret"


class DataBridgeError(RuntimeError):
    """Base class for bridge related exceptions."""


class DataBridgeConnectionError(DataBridgeError):
    """Raised when a connection to the shared bridge cannot be established."""


@dataclass
class _LineSeries:
    """Container for line chart state."""

    label: str
    points: deque


class DataBridge:
    """Thread-safe storage for frames and chart data.

    Parameters
    ----------
    max_line_points:
        Maximum number of points retained per line chart series.
    line_history_seconds:
        Soft time horizon for trimming old samples.
    use_mock_generators:
        When ``True`` a pair of background threads will emit synthetic
        telemetry and video frames. This is useful for testing the UI without
        running ``main.py``.
    video_source:
        Optional OpenCV capture index/path used by the mock generator. When the
        mock generator is disabled this value is ignored.
    chart_interval:
        Sleep duration (seconds) between synthetic chart updates.
    video_interval:
        Sleep duration (seconds) between synthetic video updates.
    """

    def __init__(
        self,
        *,
        max_line_points: int = 600,
        line_history_seconds: float = 60.0,
        use_mock_generators: bool = False,
        video_source: Optional[Any] = 0,
        chart_interval: float = 0.25,
        video_interval: float = 1.0 / 30.0,
    ) -> None:
        self._lock = threading.RLock()
        self._frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        self._bar_data: Dict[str, Dict[str, Any]] = {}
        self._line_data: Dict[str, _LineSeries] = {}
        self._max_line_points = max(30, int(max_line_points))
        self._line_history_seconds = max(1.0, float(line_history_seconds))
        self._stop_event = threading.Event()

        self._mock_enabled = False
        self._mock_threads: List[threading.Thread] = []
        self._mock_stop = threading.Event()
        self._chart_interval = max(0.05, float(chart_interval))
        self._video_interval = max(1.0 / 60.0, float(video_interval))
        self._video_source = video_source

        # Recent metadata for quick diagnostics
        self._metadata: Dict[str, Any] = {
            "created_at": time.time(),
            "mock_generators": use_mock_generators,
            "frame_resolution": (640, 480),
            "last_error": None,
        }

        if use_mock_generators:
            self.set_mock_mode(True)

    # ---------------------------- Public API ----------------------------
    def get_video_frame(self) -> np.ndarray:
        """Return the most recent frame as a ``np.ndarray`` in BGR format."""
        with self._lock:
            if self._frame is None:
                frame = self._generate_placeholder_frame("Waiting for stream...")
                self._frame = frame
                self._frame_timestamp = time.time()
            return self._frame.copy()

    def get_bar_chart_data(self) -> List[Dict[str, Any]]:
        """Return a deep copy of the bar chart payload."""
        with self._lock:
            return [dict(item) for item in self._bar_data.values()]

    def get_line_chart_data(self) -> List[Dict[str, Any]]:
        """Return the line chart payload ready for plotting."""
        payload: List[Dict[str, Any]] = []
        with self._lock:
            for series_id, series in self._line_data.items():
                points = [
                    {"timestamp": float(ts), "value": float(val)}
                    for ts, val in list(series.points)
                ]
                payload.append(
                    {
                        "id": series_id,
                        "label": series.label,
                        "points": points,
                    }
                )
        return payload

    def update_data(self, data_type: str, data: Any) -> None:
        """Update bridge storage with producer data.

        Parameters
        ----------
        data_type:
            "frame", "bar", "line", "line_points" (case insensitive).
        data:
            Payload matching the expected format.
        """

        if data_type is None:
            return
        kind = data_type.lower()
        if kind in {"frame", "video"}:
            self._update_frame(data)
        elif kind == "bar":
            self._update_bar_data(data)
        elif kind in {"line", "timeseries", "line_points"}:
            self._update_line_data(data)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

    def set_mock_mode(self, enabled: bool) -> None:
        """Enable or disable the synthetic data generators."""
        if enabled and self._mock_enabled:
            return
        if not enabled and not self._mock_enabled:
            return
        if enabled:
            self._mock_stop.clear()
            self._mock_enabled = True
            chart_thread = threading.Thread(
                target=self._mock_chart_loop,
                name="DataBridgeMockChart",
                daemon=True,
            )
            video_thread = threading.Thread(
                target=self._mock_video_loop,
                name="DataBridgeMockVideo",
                daemon=True,
            )
            self._mock_threads = [chart_thread, video_thread]
            for thread in self._mock_threads:
                thread.start()
        else:
            self._mock_enabled = False
            self._mock_stop.set()
            for thread in self._mock_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            self._mock_threads.clear()

    def get_metadata(self) -> Dict[str, Any]:
        """Return bridge metadata for diagnostics."""
        with self._lock:
            payload = dict(self._metadata)
            payload.update(
                {
                    "last_frame_timestamp": self._frame_timestamp,
                    "bar_count": len(self._bar_data),
                    "line_count": len(self._line_data),
                    "mock_enabled": self._mock_enabled,
                }
            )
            return payload

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        """Gracefully stop background activity."""
        self._stop_event.set()
        self.set_mock_mode(False)

    # ------------------------- Internal helpers -------------------------
    def _update_frame(self, frame: Any) -> None:
        if frame is None:
            return
        if not isinstance(frame, np.ndarray):
            raise TypeError("Video frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] not in (1, 3, 4):
            raise ValueError("Video frame must be HxWxC (C=1,3,4)")
        with self._lock:
            self._frame = frame.copy()
            self._frame_timestamp = time.time()
            self._metadata["frame_resolution"] = (frame.shape[1], frame.shape[0])

    def _update_bar_data(self, data: Any) -> None:
        if data is None:
            return
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, Iterable):
            items = list(data)
        else:
            raise TypeError("Bar data must be dict or iterable of dicts")
        now_ts = time.time()
        with self._lock:
            for raw in items:
                if raw is None:
                    continue
                ident = str(raw.get("id") or raw.get("label") or len(self._bar_data))
                label = str(raw.get("label") or ident)
                value = float(raw.get("value", 0.0))
                timestamp = float(raw.get("timestamp", now_ts))
                self._bar_data[ident] = {
                    "id": ident,
                    "label": label,
                    "value": value,
                    "timestamp": timestamp,
                }

    def _update_line_data(self, data: Any) -> None:
        if data is None:
            return
        payload: List[Dict[str, Any]]
        if isinstance(data, dict):
            payload = [data]
        elif isinstance(data, Iterable):
            payload = list(data)
        else:
            raise TypeError("Line data must be dict or iterable of dicts")
        with self._lock:
            for item in payload:
                if item is None:
                    continue
                series_id = str(item.get("id") or item.get("label") or len(self._line_data))
                label = str(item.get("label") or series_id)
                points = item.get("points")
                timestamp = item.get("timestamp")
                value = item.get("value")

                if series_id not in self._line_data:
                    self._line_data[series_id] = _LineSeries(
                        label=label, points=deque(maxlen=self._max_line_points)
                    )
                series = self._line_data[series_id]
                series.label = label  # keep latest label

                if points is not None:
                    for point in points:
                        ts = float(point["timestamp"])
                        val = float(point["value"])
                        series.points.append((ts, val))
                elif timestamp is not None and value is not None:
                    ts = float(timestamp)
                    val = float(value)
                    series.points.append((ts, val))
                else:
                    continue

                # Trim history based on time horizon
                cutoff = series.points[-1][0] - self._line_history_seconds
                while series.points and series.points[0][0] < cutoff:
                    series.points.popleft()

    def _mock_chart_loop(self) -> None:
        seed = secrets.randbits(32)
        while not self._stop_event.is_set() and not self._mock_stop.is_set():
            now_ts = time.time()
            phase = now_ts - math.floor(now_ts)
            bars = [
                {
                    "id": "left_elbow",
                    "label": "Left Elbow",
                    "value": 60 + 30 * math.sin(now_ts * 0.8 + seed * 0.0001),
                    "timestamp": now_ts,
                },
                {
                    "id": "left_shoulder",
                    "label": "Left Shoulder",
                    "value": 55 + 35 * math.cos(now_ts * 0.6 + seed * 0.0003),
                    "timestamp": now_ts,
                },
                {
                    "id": "right_elbow",
                    "label": "Right Elbow",
                    "value": 65 + 25 * math.sin(now_ts * 0.7 + 1.2),
                    "timestamp": now_ts,
                },
                {
                    "id": "right_shoulder",
                    "label": "Right Shoulder",
                    "value": 58 + 28 * math.cos(now_ts * 0.9 + phase),
                    "timestamp": now_ts,
                },
            ]
            lines = [
                {
                    "id": "left_wrist_speed",
                    "label": "Left Wrist Speed",
                    "timestamp": now_ts,
                    "value": 40 + 35 * math.sin(now_ts * 0.9),
                },
                {
                    "id": "right_wrist_speed",
                    "label": "Right Wrist Speed",
                    "timestamp": now_ts,
                    "value": 45 + 30 * math.cos(now_ts * 0.85 + 0.4),
                },
                {
                    "id": "left_elbow_angle",
                    "label": "Left Elbow Angle",
                    "timestamp": now_ts,
                    "value": 90 + 45 * math.sin(now_ts * 0.5 + 0.2),
                },
                {
                    "id": "right_elbow_angle",
                    "label": "Right Elbow Angle",
                    "timestamp": now_ts,
                    "value": 100 + 35 * math.cos(now_ts * 0.55 + 1.1),
                },
            ]
            try:
                self._update_bar_data(bars)
                self._update_line_data(lines)
            except Exception as exc:  # pragma: no cover - defensive guard
                with self._lock:
                    self._metadata["last_error"] = f"mock_chart_loop: {exc}"
            time.sleep(self._chart_interval)

    def _mock_video_loop(self) -> None:
        cap = None
        try:
            if self._video_source is not None:
                cap = cv2.VideoCapture(self._video_source)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as exc:  # pragma: no cover - defensive guard
            cap = None
            with self._lock:
                self._metadata["last_error"] = f"video_capture_init: {exc}"

        frame_idx = 0
        while not self._stop_event.is_set() and not self._mock_stop.is_set():
            frame_idx += 1
            frame: Optional[np.ndarray] = None
            if cap is not None and cap.isOpened():
                ok, grabbed = cap.read()
                if ok:
                    frame = grabbed
                else:
                    frame = None
            if frame is None:
                frame = self._generate_placeholder_frame(f"Demo {frame_idx:04d}")
            try:
                self._update_frame(frame)
            except Exception as exc:  # pragma: no cover - defensive guard
                with self._lock:
                    self._metadata["last_error"] = f"mock_video_loop: {exc}"
            time.sleep(self._video_interval)

        if cap is not None:
            cap.release()

    @staticmethod
    def _generate_placeholder_frame(text: str) -> np.ndarray:
        w, h = 640, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (25, 25, 25)
        cv2.putText(
            frame,
            text,
            (30, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return frame


# ---------------------------------------------------------------------------
# Multiprocessing manager glue
# ---------------------------------------------------------------------------
_bridge_singleton: Optional[DataBridge] = None
_bridge_singleton_lock = threading.Lock()


class _BridgeManager(BaseManager):
    """Custom manager exposing a singleton DataBridge."""


def _get_bridge_singleton(**kwargs: Any) -> DataBridge:
    global _bridge_singleton
    with _bridge_singleton_lock:
        if _bridge_singleton is None:
            _bridge_singleton = DataBridge(**kwargs)
        return _bridge_singleton


_BridgeManager.register(
    "DataBridge",
    callable=_get_bridge_singleton,
    exposed=[
        "get_video_frame",
        "get_bar_chart_data",
        "get_line_chart_data",
        "update_data",
        "set_mock_mode",
        "get_metadata",
        "is_running",
        "stop",
    ],
)


def _decode_authkey(value: Optional[str]) -> bytes:
    if not value:
        return _DEFAULT_AUTHKEY
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise DataBridgeConnectionError("Invalid auth key format") from exc


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        return port


def start_bridge_server(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    authkey: Optional[bytes] = None,
    **bridge_kwargs: Any,
) -> Tuple[_BridgeManager, Any, Tuple[str, int, bytes]]:
    """Launch the manager process hosting the bridge."""

    chosen_port = port or find_free_tcp_port(host)
    chosen_auth = authkey or secrets.token_bytes(16)

    manager = _BridgeManager(address=(host, chosen_port), authkey=chosen_auth)
    manager.start()
    bridge = manager.DataBridge(**bridge_kwargs)
    return manager, bridge, (host, chosen_port, chosen_auth)


def connect_to_bridge(
    host: str,
    port: int,
    authkey: bytes,
) -> Tuple[_BridgeManager, Any]:
    """Connect to an existing bridge manager and return a proxy."""

    manager = _BridgeManager(address=(host, port), authkey=authkey)
    manager.connect()
    bridge = manager.DataBridge()  # returns the singleton instance
    return manager, bridge


def configure_env_for_bridge(env: Optional[Dict[str, str]], *, host: str, port: int, authkey: bytes) -> Dict[str, str]:
    """Populate environment variables required to reach the bridge."""

    target_env = dict(os.environ if env is None else env)
    target_env[ENV_HOST] = host
    target_env[ENV_PORT] = str(port)
    target_env[ENV_AUTHKEY] = authkey.hex()
    return target_env


def connect_to_bridge_from_env(env: Optional[Dict[str, str]] = None) -> Tuple[_BridgeManager, Any]:
    """Connect to the bridge using environment variables."""

    lookup = os.environ if env is None else env
    host = lookup.get(ENV_HOST, "127.0.0.1")
    try:
        port = int(lookup.get(ENV_PORT, "0"))
    except ValueError as exc:
        raise DataBridgeConnectionError("Port must be an integer") from exc
    authkey = _decode_authkey(lookup.get(ENV_AUTHKEY))
    if port <= 0:
        raise DataBridgeConnectionError("Bridge port is missing or invalid")
    return connect_to_bridge(host, port, authkey)


__all__ = [
    "DataBridge",
    "DataBridgeError",
    "DataBridgeConnectionError",
    "start_bridge_server",
    "connect_to_bridge",
    "connect_to_bridge_from_env",
    "configure_env_for_bridge",
    "find_free_tcp_port",
    "ENV_HOST",
    "ENV_PORT",
    "ENV_AUTHKEY",
]
