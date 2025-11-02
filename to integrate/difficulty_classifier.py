import time
from collections import deque
import numpy as np
import math

# (main) from difficulty_classifier import make_difficulty_state, update_difficulty_state

# (main_initialization) diff_state = make_difficulty_state(window_sec=1.0, fps_estimate=30)

# (main_loop) mode_out, hard_score, dbg = update_difficulty_state(
# diff_state,
# R_wrist_xy=(rw_x, rw_y),R_shoulder_xy=(float(RS.x),
# float(RS.y)),
# now_t=time.time()
#)

# （main_HUD）cv2.putText(
#               frame,
#               f"Mode:{mode_out}  Score:{hard_score:.2f}",
#              (10, 100),
#               cv2.FONT_HERSHEY_SIMPLEX,
#               0.7,
#               (0,255,255) if mode_out == "easy" else (0,0,255),
#               2
#           )

def make_difficulty_state(window_sec=1.0, fps_estimate=30):
    """
    Initialize the difficulty detection state for right-hand motion analysis.

    Args:
        window_sec (float): Size of the moving window in seconds for measuring activity level.
        fps_estimate (float): Estimated frame rate (used to determine buffer length).

    Returns:
        dict: A state dictionary that stores recent wrist positions, angles, and history.
    """
    maxlen = int(window_sec * fps_estimate)
    return {
        # Motion history buffers
        "R_wrist_history": deque(maxlen=int(fps_estimate * 2)),  # Stores (t, x, y)
        "R_wrist_y": deque(maxlen=maxlen),
        "R_wrist_x": deque(maxlen=maxlen),
        "R_shoulder_angles": deque(maxlen=maxlen),

        # Overhead posture tracking
        "overhead_start_time": None,
        "overhead_hold_s": 0.0,
        "prev_overhead": False,

        # Short-term intent memory (climb / slam)
        "recent_climb_time": 0.0,
        "recent_climb_intent": 0.0,

        # Last mode and cooldown control
        "last_mode": "easy",
        "last_mode_switch_t": time.time(),
    }


def _estimate_shoulder_angle_deg(R_wrist_xy, R_shoulder_xy):
    """
    Estimate the shoulder-to-wrist elevation angle (degrees).

    Coordinate convention:
        - x increases to the right
        - y increases downward (in Mediapipe normalized image coordinates)
        - A smaller y means a higher hand position.

    The function computes the approximate upward angle between the shoulder and wrist.
    Larger angles correspond to a raised arm.

    Returns:
        float: Shoulder angle in degrees within [0, 180].
    """
    if (R_wrist_xy is None) or (R_shoulder_xy is None):
        return 0.0

    wx, wy = R_wrist_xy
    sx, sy = R_shoulder_xy

    dx = wx - sx
    dy = wy - sy  # Smaller y means higher position

    # Convert to an angle: higher wrist (smaller y) → larger positive angle
    ang_rad = math.atan2(-(dy), abs(dx) + 1e-9)
    ang_deg = math.degrees(ang_rad)

    # Clip angle to [0, 180]
    return float(max(0.0, min(180.0, ang_deg)))



def update_difficulty_state(
    state,
    R_wrist_xy,
    R_shoulder_xy,
    now_t=None,
):
    """
    Update and evaluate the difficulty mode based on right-hand posture and motion.

    Args:
        state (dict): The current difficulty state dictionary.
        R_wrist_xy (tuple): (x, y) coordinates of the right wrist (normalized 0–1).
        R_shoulder_xy (tuple): (x, y) coordinates of the right shoulder (normalized 0–1).
        now_t (float): Current timestamp (defaults to time.time()).

    Returns:
        tuple:
            mode_out (str): "easy" or "hard"
            hard_score (float): Continuous score [0, 1] representing difficulty
            debug_info (dict): All intermediate metrics for debugging or visualization
    """
    if now_t is None:
        now_t = time.time()

    # -------- 0. Input defaults --------
    if R_wrist_xy is None:
        R_wrist_xy = (0.5, 0.5)
    if R_shoulder_xy is None:
        R_shoulder_xy = (0.5, 0.6)

    rw_x, rw_y = R_wrist_xy
    rs_x, rs_y = R_shoulder_xy

    # Compute shoulder elevation angle
    R_shoulder_angle = _estimate_shoulder_angle_deg(
        R_wrist_xy=R_wrist_xy,
        R_shoulder_xy=R_shoulder_xy,
    )

    # -------- 1. Record wrist and shoulder history --------
    state["R_wrist_history"].append((now_t, rw_x, rw_y))
    state["R_wrist_y"].append(rw_y)
    state["R_wrist_x"].append(rw_x)
    state["R_shoulder_angles"].append(R_shoulder_angle)

    # -------- 2. Detect "overhead" posture --------
    HAND_HIGH_Y = 0.35           # Smaller y = higher position
    SHOULDER_OPEN_ANGLE = 100.0  # Threshold for raised arm
    is_overhead_now = (rw_y < HAND_HIGH_Y) or (R_shoulder_angle > SHOULDER_OPEN_ANGLE)

    # -------- 3. Track overhead hold duration --------
    if is_overhead_now:
        if state["prev_overhead"]:
            state["overhead_hold_s"] = now_t - state["overhead_start_time"]
        else:
            state["overhead_start_time"] = now_t
            state["overhead_hold_s"] = 0.0
    else:
        state["overhead_hold_s"] = 0.0
        state["overhead_start_time"] = now_t
    state["prev_overhead"] = is_overhead_now
    overhead_hold_s = state["overhead_hold_s"]

    # -------- 4. Compute activity level (motion jitter) --------
    activity_level = 0.0
    if len(state["R_wrist_x"]) > 2:
        xs = np.array(state["R_wrist_x"])
        ys = np.array(state["R_wrist_y"])
        activity_level = float(np.std(xs) + np.std(ys))

    # -------- 5. Detect "climb" motion (high → downward sweep) --------
    climb_event = False
    climb_drop_thresh = 0.05
    climb_window = 0.6
    HAND_HIGH_Y = 0.35

    for t0, _, y0 in reversed(state["R_wrist_history"]):
        if now_t - t0 > climb_window:
            break
        if y0 < HAND_HIGH_Y and (rw_y - y0) > climb_drop_thresh:
            climb_event = True
            break

    # -------- 6. Detect "slam" motion (mid → fast downward) --------
    slam_event = False
    slam_drop_thresh = 0.12
    slam_window = 0.10
    mid_low, mid_high = 0.35, 0.65

    for t0, _, y0 in reversed(state["R_wrist_history"]):
        dt = now_t - t0
        if dt > slam_window:
            break
        if mid_low <= y0 <= mid_high and (rw_y - y0) >= slam_drop_thresh:
            slam_event = True
            break

    # -------- 7. Update "recent intent" memory (climb/slam decay) --------
    if climb_event or slam_event:
        state["recent_climb_time"] = now_t
        state["recent_climb_intent"] = 1.0

    # Linearly decay over 1.5 seconds
    dt_since_climb = now_t - state["recent_climb_time"]
    decay_window = 1.5
    recent_climb_intent = max(0.0, 1.0 - dt_since_climb / decay_window)
    state["recent_climb_intent"] = recent_climb_intent

    # -------- 8. Normalize key features --------
    overhead_level = 1.0 if is_overhead_now else 0.0
    stability_level = float(np.clip(1.0 - activity_level / 0.1, 0.0, 1.0))
    overhead_hold_norm = min(overhead_hold_s / 2.0, 1.0)
    climb_intent_norm = recent_climb_intent

    # -------- 9. Weighted hard score --------
    w1, w2, w3, w4 = 0.30, 0.25, 0.15, 0.30
    hard_score = (
        w1 * overhead_level +
        w2 * stability_level +
        w3 * overhead_hold_norm +
        w4 * climb_intent_norm
    )
    hard_score = float(np.clip(hard_score, 0.0, 1.0))

    # -------- 10. Determine difficulty mode with cooldown --------
    HARD_THRESHOLD = 0.4
    mode_candidate = "hard" if hard_score >= HARD_THRESHOLD else "easy"
    COOLDOWN = 0.3

    if mode_candidate != state["last_mode"]:
        if (now_t - state["last_mode_switch_t"]) > COOLDOWN:
            state["last_mode"] = mode_candidate
            state["last_mode_switch_t"] = now_t

    mode_out = state["last_mode"]

    # -------- 11. Build debug info --------
    debug_info = {
        "hard_score": hard_score,
        "overhead_hold_s": overhead_hold_s,
        "activity_level": activity_level,
        "recent_climb_intent": recent_climb_intent,
        "climb_event": climb_event,
        "slam_event": slam_event,
        "overhead_level": overhead_level,
        "stability_level": stability_level,
        "R_shoulder_angle_deg": R_shoulder_angle,
        "rw_x": rw_x,
        "rw_y": rw_y,
        "rs_x": rs_x,
        "rs_y": rs_y,
    }

    return mode_out, hard_score, debug_info

"""How can we represent the two modes in the sessio_summary figures?"""
"""I'm thinking it might be better to use two different colors for the lines in the summary figure 
to show the two modes separately."""
