"""Streamlit-based real-time monitoring dashboard.

The dashboard is split into a 40/60 column layout. The left column displays
four live-updating bar charts and two line charts, while the right column embeds
an OpenCV video feed at 640x480 resolution.

Data is retrieved from :mod:`data_bridge` which exposes a multiprocessing-aware
``DataBridge`` instance. The Streamlit script connects to the bridge using the
``DATA_BRIDGE_*`` environment variables so it can run as a standalone process or
side-by-side with ``main.py`` when launched via ``ui_launcher.py``.
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data_bridge import (
    DataBridgeConnectionError,
    ENV_AUTHKEY,
    ENV_HOST,
    ENV_PORT,
    connect_to_bridge_from_env,
)

matplotlib.use("Agg")

FRAME_INTERVAL = 1.0 / 30.0
CHART_INTERVAL = 0.3
BAR_SERIES = [
    ("left_elbow", "Left Elbow"),
    ("left_shoulder", "Left Shoulder"),
    ("right_elbow", "Right Elbow"),
    ("right_shoulder", "Right Shoulder"),
]
LINE_GROUPS = [
    {
        "title": "Wrist Speed",
        "ylabel": "Speed",
        "series": [
            {"id": "left_wrist_speed", "label": "Left Wrist Speed"},
            {"id": "right_wrist_speed", "label": "Right Wrist Speed"},
        ],
    },
    {
        "title": "Elbow Angle",
        "ylabel": "Angle (deg)",
        "series": [
            {"id": "left_elbow_angle", "label": "Left Elbow Angle"},
            {"id": "right_elbow_angle", "label": "Right Elbow Angle"},
        ],
    },
]
FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans CN",
    "Microsoft YaHei",
    "PingFang SC",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]


def _configure_matplotlib_fonts() -> None:
    if getattr(_configure_matplotlib_fonts, "_configured", False):
        return
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for candidate in FONT_CANDIDATES:
        if candidate in available_fonts:
            chosen = candidate
            break
    fallback_stack = [font for font in ([chosen] if chosen else []) + ["DejaVu Sans", "sans-serif"]]
    matplotlib.rcParams["font.sans-serif"] = fallback_stack
    matplotlib.rcParams["axes.unicode_minus"] = False
    _configure_matplotlib_fonts._configured = True


@st.cache_data(show_spinner=False)
def _load_custom_css() -> str:
    return """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.0rem;
    }
    .dashboard-panel {
        width: 100%;
        box-sizing: border-box;
        border-radius: 12px;
        border: 1px solid rgba(30, 64, 175, 0.65);
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        padding: 1rem 1.1rem 1.2rem 1.1rem;
        margin: 0;
    }
    .dashboard-panel.panel-left {
        border: 1px solid rgba(31, 41, 55, 0.75);
        background: linear-gradient(180deg, #101828 0%, #0b1220 100%);
    }
    .dashboard-panel :where(figure, img) {
        margin: 0 !important;
        max-width: 100%;
    }
    .dashboard-panel :where(figure, img) {
        display: block;
    }
    .dashboard-panel .stImage > img {
        border-radius: 8px;
    }
    .dashboard-panel .element-container,
    .dashboard-panel [data-testid="stVerticalBlock"],
    .dashboard-panel [data-testid="stHorizontalBlock"],
    .dashboard-panel [data-testid="column"] > div,
    .dashboard-panel [data-testid="column"] > div > div {
        width: 100%;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    .dashboard-panel [data-testid="column"] {
        gap: 0.75rem !important;
    }
    .dashboard-panel .stMarkdown,
    .dashboard-panel .stPlot,
    .dashboard-panel .stImage,
    .dashboard-panel .stAltairChart,
    .dashboard-panel .stPlotlyChart {
        width: 100%;
    }
    </style>
    """


def _ensure_bridge_connection() -> Tuple[Any, Any]:
    if "bridge" not in st.session_state:
        try:
            manager, bridge = connect_to_bridge_from_env()
        except DataBridgeConnectionError as exc:
            st.error(
                "Unable to connect to DataBridge. Make sure `ui_launcher.py` is running and environment variables are configured." f"\nError: {exc}"
            )
            st.stop()
        st.session_state.bridge = bridge
        st.session_state.bridge_manager = manager
    return st.session_state.bridge_manager, st.session_state.bridge


def _build_left_chart_grid() -> Tuple[List[st.delta_generator.DeltaGenerator], List[st.delta_generator.DeltaGenerator]]:
    bar_placeholders: List[st.delta_generator.DeltaGenerator] = []
    line_placeholders: List[st.delta_generator.DeltaGenerator] = []
    row1_col1, row1_col2 = st.columns(2, gap="medium")
    row2_col1, row2_col2 = st.columns(2, gap="medium")
    row3_col1, row3_col2 = st.columns(2, gap="medium")
    for col in (row1_col1, row1_col2, row2_col1, row2_col2):
        bar_placeholders.append(col.empty())
    for col in (row3_col1, row3_col2):
        line_placeholders.append(col.empty())
    return bar_placeholders, line_placeholders


def _prepare_bar_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now_ts = time.time()
    indexed = {item.get("id"): item for item in data}
    prepared: List[Dict[str, Any]] = []
    for series_id, label in BAR_SERIES:
        entry = dict(indexed.get(series_id, {}))
        if not entry:
            entry = {"id": series_id, "label": label, "value": 0.0, "timestamp": now_ts}
        entry.setdefault("label", label)
        entry.setdefault("timestamp", now_ts)
        prepared.append(entry)
    return prepared


def _prepare_line_groups(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed = {item.get("id"): item for item in data}
    groups: List[Dict[str, Any]] = []
    now_ts = time.time()
    for group in LINE_GROUPS:
        series_payload: List[Dict[str, Any]] = []
        for spec in group["series"]:
            entry = dict(indexed.get(spec["id"], {}))
            if not entry:
                entry = {
                    "id": spec["id"],
                    "label": spec["label"],
                    "points": [],
                    "timestamp": now_ts,
                }
            entry.setdefault("label", spec["label"])
            entry_points = entry.get("points") or []
            entry["points"] = entry_points
            series_payload.append(entry)
        groups.append(
            {
                "title": group["title"],
                "ylabel": group["ylabel"],
                "series": series_payload,
            }
        )
    return groups


def _render_bar_chart(placeholder, label: str, value: float) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.bar([label], [value], color="#4f46e5")
    ax.set_ylim(0, max(100, value * 1.2))
    ax.set_ylabel("Value")
    ax.set_title(label, fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    placeholder.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)


def _render_line_chart(
    placeholder,
    title: str,
    series_items: List[Dict[str, Any]],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    palette = ["#22d3ee", "#f97316", "#8b5cf6", "#14b8a6", "#facc15"]
    plotted = False
    for idx, item in enumerate(series_items):
        points = item.get("points", [])
        if not points:
            continue
        xs = [datetime.fromtimestamp(pt["timestamp"]) for pt in points]
        ys = [pt["value"] for pt in points]
        color = palette[idx % len(palette)]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=item.get("label", f"Series {idx+1}"))
        plotted = True
    if not plotted:
        ax.plot([], [])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    if plotted:
        ax.legend(loc="upper left")
    fig.autofmt_xdate()
    ax.grid(alpha=0.4, linestyle="--")
    fig.tight_layout()
    placeholder.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)


def _render_video_frame(placeholder, frame: np.ndarray) -> None:
    if frame.ndim == 2 or frame.shape[2] == 1:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder.image(display, caption="Live Stream (640x480)", use_column_width=True)


def _render_status(status_placeholder, bridge) -> None:
    try:
        meta = bridge.get_metadata()
    except Exception as exc:  # pragma: no cover - defensive guard
        status_placeholder.warning(f"Can't read bridge status: {exc}")
        return
    ts = meta.get("last_frame_timestamp")
    ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "--"
    status_placeholder.info(
        f"Bridge Status | mock={meta.get('mock_enabled')} | "
        f"bar={meta.get('bar_count')} line={meta.get('line_count')} | "
        f"Last Frame Time={ts_str}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Live Monitoring Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_load_custom_css(), unsafe_allow_html=True)
    _configure_matplotlib_fonts()

    manager, bridge = _ensure_bridge_connection()

    if "monitor_running" not in st.session_state:
        st.session_state.monitor_running = True

    st.title("Live Monitoring Dashboard")
    info_placeholder = st.empty()

    with st.sidebar:
        st.markdown("### Controls")
        if st.button("Pause/Resume Refresh"):
            st.session_state.monitor_running = not st.session_state.monitor_running
        st.caption(
            "Environment Variables: "
            f"``{ENV_HOST}``={os.environ.get(ENV_HOST, '127.0.0.1')} · "
            f"``{ENV_PORT}``={os.environ.get(ENV_PORT, 'unset')} · "
            f"``{ENV_AUTHKEY}``={'***' if ENV_AUTHKEY in os.environ else 'unset'}"
        )

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        st.subheader("Joint Metrics", divider="rainbow")
        with st.container():
            st.markdown('<div class="dashboard-panel panel-left">', unsafe_allow_html=True)
            bar_placeholders, line_placeholders = _build_left_chart_grid()
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("Video Stream", divider="rainbow")
        with st.container():
            st.markdown('<div class="dashboard-panel panel-right">', unsafe_allow_html=True)
            video_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

    last_chart_update = 0.0
    last_frame_update = 0.0

    try:
        while True:
            loop_now = time.perf_counter()
            if st.session_state.monitor_running and (loop_now - last_frame_update) >= FRAME_INTERVAL:
                try:
                    frame = bridge.get_video_frame()
                except Exception as exc:
                    info_placeholder.warning(f"Failed to read video stream: {exc}")
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _render_video_frame(video_placeholder, frame)
                last_frame_update = loop_now

            if st.session_state.monitor_running and (loop_now - last_chart_update) >= CHART_INTERVAL:
                try:
                    bars = bridge.get_bar_chart_data()
                    lines = bridge.get_line_chart_data()
                except Exception as exc:
                    info_placeholder.error(f"Failed to read chart data: {exc}")
                    bars, lines = [], []

                bar_items = _prepare_bar_items(bars)
                for placeholder, bar_item in zip(bar_placeholders, bar_items):
                    _render_bar_chart(
                        placeholder,
                        bar_item.get("label", "Bar Chart"),
                        float(bar_item.get("value", 0.0)),
                    )

                line_groups = _prepare_line_groups(lines)
                for placeholder, group in zip(line_placeholders, line_groups):
                    _render_line_chart(
                        placeholder,
                        group.get("title", "Time Series"),
                        list(group.get("series", [])),
                        group.get("ylabel", "Value"),
                    )

                _render_status(info_placeholder, bridge)
                last_chart_update = loop_now

            if not st.session_state.monitor_running:
                time.sleep(0.1)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:  # pragma: no cover - user initiated
        pass
    finally:
        if "bridge" in st.session_state:
            manager = st.session_state.get("bridge_manager")
            if manager is not None:
                # Keep manager alive until Streamlit session ends
                pass


if __name__ == "__main__":
    main()
