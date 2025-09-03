"""
Arm kinematics angles and displacement-rate (speed) visualizations.

This module computes basic arm angles from MediaPipe Pose landmarks and tracks
wrist displacement rate (speed). It also renders:
- A bar chart of current angles (left/right elbow, left/right shoulder).
- Time-series line charts for speeds and angles over a recent time window.

Dependencies: numpy, opencv-python (cv2). MediaPipe is optional; if available,
`extract_points_from_mediapipe` can adapt pose results directly.

Typical integration (non-invasive) with your main loop:

    import time
    import cv2
    from arm_kinematics_viz import ArmKinematicsVisualizer, extract_points_from_mediapipe

    viz = ArmKinematicsVisualizer(history_sec=6.0)

    # inside your loop after getting res_pose (MediaPipe Pose) and normalized
    # wrist coordinates rw_x, rw_y, lw_x, lw_y (0..1), plus current time now_t:
    points = extract_points_from_mediapipe(res_pose)  # returns dict or None
    panel  = viz.update_and_render(points=points,
                                   now=time.time(),
                                   rw=(rw_x, rw_y) if rw_x is not None else None,
                                   lw=(lw_x, lw_y) if lw_x is not None else None)
    if panel is not None:
        cv2.imshow('Arm Kinematics', panel)

All functions are robust to missing data; None values are skipped gracefully.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2

# Optional import for convenience adapter
try:
    import mediapipe as mp
    _HAVE_MP = True
except Exception:
    mp = None
    _HAVE_MP = False

Point = Tuple[float, float]

# ---- Geometry helpers ----

def _safe_vec(a: Point, b: Point) -> np.ndarray:
    return np.array([b[0]-a[0], b[1]-a[1]], dtype=np.float64)


def compute_angle(a: Point, b: Point, c: Point) -> Optional[float]:
    """Angle ABC in degrees using 2D points.
    Returns None if vectors are degenerate.
    """
    ab = _safe_vec(b, a)  # from B to A
    cb = _safe_vec(b, c)  # from B to C
    nab = np.linalg.norm(ab)
    ncb = np.linalg.norm(cb)
    if nab < 1e-9 or ncb < 1e-9:
        return None
    cosang = float(np.clip(np.dot(ab, cb) / (nab * ncb), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


# ---- Pose adapters and angle computations ----

POSE_KEYS = {
    'L_SHOULDER': 11,
    'R_SHOULDER': 12,
    'L_ELBOW': 13,
    'R_ELBOW': 14,
    'L_WRIST': 15,
    'R_WRIST': 16,
    'L_HIP': 23,
    'R_HIP': 24,
}


def extract_points_from_mediapipe(res_pose) -> Optional[Dict[str, Point]]:
    """Extract normalized (x,y) points for needed landmarks from MediaPipe Pose results.
    Returns dict mapping keys in POSE_KEYS to (x, y) or None if unavailable.
    """
    if not _HAVE_MP:
        return None
    if res_pose is None or res_pose.pose_landmarks is None:
        return None
    lm = res_pose.pose_landmarks.landmark
    pts = {}
    try:
        for k, idx in POSE_KEYS.items():
            p = lm[idx]
            # Use normalized coordinates (0..1). Visibility guard optional.
            if getattr(p, 'visibility', 1.0) < 0.2:
                pts[k] = None
            else:
                pts[k] = (float(p.x), float(p.y))
    except Exception:
        return None
    return pts


def get_arm_angles(points: Dict[str, Optional[Point]]) -> Dict[str, Optional[float]]:
    """Compute elbow and shoulder angles (degrees) for both sides.
    Angles:
      - L_elbow: angle at elbow formed by SHOULDER-ELBOW-WRIST
      - R_elbow: angle at elbow formed by SHOULDER-ELBOW-WRIST
      - L_shoulder: angle at shoulder formed by ELBOW-SHOULDER-HIP
      - R_shoulder: angle at shoulder formed by ELBOW-SHOULDER-HIP
    Returns a dict with keys above; values may be None if insufficient points.
    """
    res: Dict[str, Optional[float]] = {
        'L_elbow': None, 'R_elbow': None, 'L_shoulder': None, 'R_shoulder': None
    }
    Ls, Le, Lw, Lh = points.get('L_SHOULDER'), points.get('L_ELBOW'), points.get('L_WRIST'), points.get('L_HIP')
    Rs, Re, Rw, Rh = points.get('R_SHOULDER'), points.get('R_ELBOW'), points.get('R_WRIST'), points.get('R_HIP')
    if Ls and Le and Lw:
        res['L_elbow'] = compute_angle(Ls, Le, Lw)
    if Rs and Re and Rw:
        res['R_elbow'] = compute_angle(Rs, Re, Rw)
    if Le and Ls and Lh:
        res['L_shoulder'] = compute_angle(Le, Ls, Lh)
    if Re and Rs and Rh:
        res['R_shoulder'] = compute_angle(Re, Rs, Rh)
    return res


# ---- Kinematics tracker (speed of wrists) ----

@dataclass
class _Hist:
    max_sec: float
    data: deque = field(default_factory=deque)

    def add(self, t: float, x: float, y: float) -> None:
        self.data.append((t, float(x), float(y)))
        t_cut = t - self.max_sec
        while self.data and self.data[0][0] < t_cut:
            self.data.popleft()

    def velocity(self) -> Optional[Tuple[float, float]]:
        n = len(self.data)
        if n < 2:
            return None
        # Weighted least squares slope for robustness akin to main.py
        ts = np.array([p[0] for p in self.data], dtype=np.float64)
        xs = np.array([p[1] for p in self.data], dtype=np.float64)
        ys = np.array([p[2] for p in self.data], dtype=np.float64)
        A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
        try:
            ax, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
            ay, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            dt = max(ts[-1] - ts[0], 1e-6)
            ax = (xs[-1] - xs[0]) / dt
            ay = (ys[-1] - ys[0]) / dt
        return float(ax), float(ay)

    def speed(self) -> Optional[float]:
        v = self.velocity()
        return None if v is None else float(math.hypot(v[0], v[1]))


class KinematicsTracker:
    """Tracks left/right wrist speeds over a sliding time window.
    Coordinates are expected in the same space/frame over time (e.g., normalized 0..1).
    """
    def __init__(self, history_sec: float = 0.8):
        self.left = _Hist(history_sec)
        self.right = _Hist(history_sec)

    def update(self, now: float, lw: Optional[Point], rw: Optional[Point]) -> None:
        if lw is not None:
            self.left.add(now, lw[0], lw[1])
        if rw is not None:
            self.right.add(now, rw[0], rw[1])

    def speeds(self) -> Dict[str, Optional[float]]:
        return {
            'L_speed': self.left.speed(),
            'R_speed': self.right.speed(),
        }


# ---- Simple renderers using OpenCV ----

def _draw_text(img, text: str, org: Tuple[int, int], color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_angle_bars(angles: Dict[str, Optional[float]], size=(480, 220)) -> np.ndarray:
    """Render a bar chart for provided angles (degrees 0..180)."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_text(img, 'Arm Angles (deg)', (10, 18), (0, 255, 255), 0.6, 2)

    keys = ['L_elbow', 'R_elbow', 'L_shoulder', 'R_shoulder']
    labels = ['L-ELB', 'R-ELB', 'L-SHO', 'R-SHO']
    n = len(keys)
    margin = 20
    bar_w = (w - 2 * margin) // n
    max_deg = 180.0
    for i, (k, lab) in enumerate(zip(keys, labels)):
        x0 = margin + i * bar_w + 8
        x1 = x0 + bar_w - 16
        y0 = h - 24
        y1 = 30
        cv2.rectangle(img, (x0, y1), (x1, y0), (50, 50, 50), 1)
        val = angles.get(k, None)
        if val is None or not (val == val):  # None or NaN
            _draw_text(img, f'{lab}: --', (x0, y0 + 16), (180, 180, 180))
            continue
        v = float(np.clip(val, 0.0, max_deg))
        fill_h = int((v / max_deg) * (y0 - y1))
        cv2.rectangle(img, (x0 + 1, y0 - fill_h), (x1 - 1, y0 - 1), (0, 200, 255), -1)
        _draw_text(img, f'{lab}:{v:5.1f}', (x0, y0 + 16), (200, 255, 200))
    return img


def render_line_chart(history: List[Tuple[float, float]],
                      size=(480, 160),
                      title: str = 'Metric',
                      y_range: Tuple[float, float] = (0.0, 1.0),
                      color=(100, 255, 100)) -> np.ndarray:
    """Render a simple time-series line chart.
    history: list of (t, value), t in seconds.
    y_range: (min, max) for scaling.
    """
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_text(img, title, (10, 18), (255, 255, 0), 0.55, 1)
    if not history:
        return img
    t0 = history[0][0]
    t1 = history[-1][0]
    if t1 <= t0:
        return img
    # Axes
    cv2.rectangle(img, (40, 24), (w - 10, h - 24), (60, 60, 60), 1)
    xmin, xmax = t0, t1
    ymin, ymax = y_range
    W = (w - 50)
    H = (h - 48)
    def to_px(t, v):
        x = 40 + int((t - xmin) / (xmax - xmin) * W)
        y = (h - 24) - int((v - ymin) / (ymax - ymin + 1e-9) * H)
        return x, y
    pts = []
    for t, v in history:
        v = float(np.clip(v, ymin, ymax))
        pts.append(to_px(t, v))
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    return img


class MetricsBuffer:
    """A rolling metrics buffer for multiple keys with fixed history seconds."""
    def __init__(self, keys: List[str], history_sec: float = 6.0):
        self.history_sec = float(history_sec)
        self.buffers: Dict[str, deque] = {k: deque() for k in keys}

    def add(self, now: float, **kv):
        for k, v in kv.items():
            if k not in self.buffers:
                self.buffers[k] = deque()
            self.buffers[k].append((now, v))
        t_cut = now - self.history_sec
        for q in self.buffers.values():
            while q and q[0][0] < t_cut:
                q.popleft()

    def series(self, key: str) -> List[Tuple[float, float]]:
        return list(self.buffers.get(key, []))


class ArmKinematicsVisualizer:
    """High-level utility to compute angles and speeds and render a composite panel.

    - Call update_and_render(points, now, lw, rw) on each frame.
    - points: dict from extract_points_from_mediapipe or similarly structured.
    - lw, rw: optional (x,y) normalized wrist positions for left/right.

    Returns an OpenCV BGR image with charts (angles bar + line graphs). If inputs
    are insufficient it still returns an empty/annotated panel.
    """
    def __init__(self, history_sec: float = 6.0,
                 panel_size: Tuple[int, int] = (640, 420)):
        self.kin = KinematicsTracker(history_sec=min(1.2, history_sec))
        self.metrics = MetricsBuffer(
            keys=['L_speed', 'R_speed', 'L_elbow', 'R_elbow', 'L_shoulder', 'R_shoulder'],
            history_sec=history_sec
        )
        self.panel_w, self.panel_h = panel_size

    def update_and_render(self,
                          points: Optional[Dict[str, Optional[Point]]],
                          now: Optional[float] = None,
                          lw: Optional[Point] = None,
                          rw: Optional[Point] = None) -> np.ndarray:
        now = time.time() if now is None else float(now)
        # Update speeds
        self.kin.update(now, lw, rw)
        spd = self.kin.speeds()
        # Angles
        ang = {'L_elbow': None, 'R_elbow': None, 'L_shoulder': None, 'R_shoulder': None}
        if points:
            ang = get_arm_angles(points)
        # Update metrics buffer (use NaN for missing values to keep spacing)
        kv = {k: (np.nan if v is None else float(v)) for k, v in {**spd, **ang}.items()}
        self.metrics.add(now, **kv)

        # Compose panel: top = angle bars; bottom split: speeds + elbow angles time-series
        top = render_angle_bars(ang, size=(self.panel_w, max(160, self.panel_h//3)))

        # Speeds time-series (auto y-range small; normalized speed ~0..maybe 1.5)
        ls = self.metrics.series('L_speed')
        rs = self.metrics.series('R_speed')
        # Build unified y-range from both series ignoring NaN
        def _clean(series):
            return [(t, float(v)) for (t, v) in series if v == v]
        ls_c, rs_c = _clean(ls), _clean(rs)
        if ls_c or rs_c:
            vals = [v for _, v in (ls_c + rs_c)]
            vmin, vmax = float(min(vals)), float(max(vals))
            pad = max(0.05, 0.15 * (vmax - vmin + 1e-6))
            y_rng = (max(0.0, vmin - pad), max(0.2, vmax + pad))
        else:
            y_rng = (0.0, 0.5)
        speed_h = max(120, (self.panel_h - top.shape[0]) // 2)
        sp_img = np.zeros((speed_h, self.panel_w, 3), dtype=np.uint8)
        l_img = render_line_chart(ls_c, size=(self.panel_w, speed_h), title='Left Wrist Speed', y_range=y_rng, color=(100, 220, 255))
        r_img = render_line_chart(rs_c, size=(self.panel_w, speed_h), title='Right Wrist Speed', y_range=y_rng, color=(100, 255, 100))
        # Blend two charts with simple max to show both
        sp_img = np.maximum(l_img, r_img)

        # Elbow angles time-series
        le = _clean(self.metrics.series('L_elbow'))
        re = _clean(self.metrics.series('R_elbow'))
        ang_rng = (40.0, 180.0)
        ang_h = self.panel_h - top.shape[0] - sp_img.shape[0]
        ang_h = max(100, ang_h)
        le_img = render_line_chart(le, size=(self.panel_w, ang_h), title='Elbow Angle (deg)', y_range=ang_rng, color=(255, 180, 100))
        re_img = render_line_chart(re, size=(self.panel_w, ang_h), title='Elbow Angle (deg)', y_range=ang_rng, color=(255, 100, 100))
        ang_img = np.maximum(le_img, re_img)

        panel = np.vstack([top, sp_img, ang_img])
        # If panel height exceeds desired, crop; else pad
        if panel.shape[0] > self.panel_h:
            panel = panel[:self.panel_h, :self.panel_w]
        elif panel.shape[0] < self.panel_h:
            pad_h = self.panel_h - panel.shape[0]
            panel = np.vstack([panel, np.zeros((pad_h, self.panel_w, 3), dtype=np.uint8)])
        return panel
