import time
from collections import deque
import numpy as np

def make_difficulty_state(window_sec=1.0, fps_estimate=30):
    """
    Initialize the difficulty detection state for right-hand control.
    window_sec: window size (seconds) for measuring activity level.
    fps_estimate: estimated frame rate (used to set buffer size).
    """
    maxlen = int(window_sec * fps_estimate)
    return {
        "R_wrist_history": deque(maxlen=int(fps_estimate * 2)),  # store (t, x, y)
        "R_wrist_y": deque(maxlen=maxlen),
        "R_wrist_x": deque(maxlen=maxlen),
        "R_shoulder_angles": deque(maxlen=maxlen),

        # Overhead hold tracking
        "overhead_start_time": None,
        "overhead_hold_s": 0.0,
        "prev_overhead": False,

        # Recent climb/slam intent memory
        "recent_climb_time": 0.0,
        "recent_climb_intent": 0.0,

        # Last output mode and cooldown
        "last_mode": "easy",
        "last_mode_switch_t": time.time(),
    }

def update_difficulty_state(
    state,
    R_shoulder_angle,
    R_y,
    R_x,
    now_t=None,
):
    """
    Analyze right-hand motion to determine whether the user is in EASY or HARD mode.
    Includes detection for climb (high→downward sweep) and slam (mid→fast downward thrust).
    Returns:
      mode_out: "easy" or "hard"
      hard_score: continuous [0,1] weighted difficulty score
      debug_info: dictionary with all intermediate metrics
    """
    if now_t is None:
        now_t = time.time()

    # ---------- 1. Append new frame data ----------
    if R_y is None:
        R_y = 0.5
    if R_x is None:
        R_x = 0.5
    if R_shoulder_angle is None:
        R_shoulder_angle = 0.0

    state["R_wrist_history"].append((now_t, R_x, R_y))
    state["R_wrist_y"].append(R_y)
    state["R_wrist_x"].append(R_x)
    state["R_shoulder_angles"].append(R_shoulder_angle)

    # ---------- 2. Determine if the hand is in overhead posture ----------
    HAND_HIGH_Y = 0.35          # smaller y = higher hand position
    SHOULDER_OPEN_ANGLE = 100.0 # degree threshold for overhead
    is_overhead_now = (R_y < HAND_HIGH_Y) or (R_shoulder_angle > SHOULDER_OPEN_ANGLE)

    # ---------- 3. Update overhead hold duration ----------
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

    # ---------- 4. Compute activity level (movement jitter) ----------
    activity_level = 0.0
    if len(state["R_wrist_x"]) > 2:
        xs = np.array(state["R_wrist_x"])
        ys = np.array(state["R_wrist_y"])
        activity_level = float(np.std(xs) + np.std(ys))

    # ---------- 5. Detect climb event (high→downward sweep) ----------
    climb_event = False
    climb_drop_thresh = 0.12  # required downward motion in y
    climb_window = 0.3        # time window for detecting downward sweep

    # Use wrist history to find previous high position
    for t0, _, y0 in reversed(state["R_wrist_history"]):
        if now_t - t0 > climb_window:
            break
        # If previously high and now much lower
        if y0 < HAND_HIGH_Y and (R_y - y0) > climb_drop_thresh:
            climb_event = True
            break

    # ---------- 6. Detect slam event (mid→fast downward motion) ----------
    slam_event = False
    slam_drop_thresh = 0.12   # required downward change
    slam_window = 0.15        # must occur in short time
    mid_low, mid_high = 0.35, 0.65

    for t0, _, y0 in reversed(state["R_wrist_history"]):
        dt = now_t - t0
        if dt > slam_window:
            break
        if mid_low <= y0 <= mid_high and (R_y - y0) >= slam_drop_thresh:
            slam_event = True
            break

    # ---------- 7. Update climb/slam memory ----------
    if climb_event or slam_event:
        state["recent_climb_time"] = now_t
        state["recent_climb_intent"] = 1.0

    # Linear decay for 1.5 seconds
    dt_since_climb = now_t - state["recent_climb_time"]
    decay_window = 1.5
    recent_climb_intent = max(0.0, 1.0 - dt_since_climb / decay_window)
    state["recent_climb_intent"] = recent_climb_intent

    # ---------- 8. Compute normalized features ----------
    overhead_level = 1.0 if is_overhead_now else 0.0
    stability_level = float(np.clip(1.0 - activity_level / 0.1, 0.0, 1.0))
    overhead_hold_norm = min(overhead_hold_s / 2.0, 1.0)
    climb_intent_norm = recent_climb_intent

    # ---------- 9. Compute weighted hard score ----------
    w1, w2, w3, w4 = 0.30, 0.25, 0.25, 0.20
    hard_score = (
        w1 * overhead_level +
        w2 * stability_level +
        w3 * overhead_hold_norm +
        w4 * climb_intent_norm
    )
    hard_score = float(np.clip(hard_score, 0.0, 1.0))

    # ---------- 10. Determine mode with cooldown ----------
    HARD_THRESHOLD = 0.5
    mode_candidate = "hard" if hard_score >= HARD_THRESHOLD else "easy"

    COOLDOWN = 0.3
    if mode_candidate != state["last_mode"]:
        if (now_t - state["last_mode_switch_t"]) > COOLDOWN:
            state["last_mode"] = mode_candidate
            state["last_mode_switch_t"] = now_t
    mode_out = state["last_mode"]

    # ---------- 11. Build debug info ----------
    debug_info = {
        "hard_score": hard_score,
        "overhead_hold_s": overhead_hold_s,
        "activity_level": activity_level,
        "recent_climb_intent": recent_climb_intent,
        "climb_event": climb_event,
        "slam_event": slam_event,
        "overhead_level": overhead_level,
        "stability_level": stability_level,
    }

    return mode_out, hard_score, debug_info

