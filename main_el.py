import time, math
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
# Optional import vgamepad: does not exist on macOS, falls back to a null implementation
try:
    from vgamepad import VX360Gamepad as _VX360Gamepad
    _GAMEPAD_AVAILABLE = True
except Exception:
    _GAMEPAD_AVAILABLE = False
    class _VX360Gamepad:
        def __init__(self): pass
        def left_joystick(self, x_value=0, y_value=0): pass
        def update(self): pass
    print("[Warning] vgamepad is not available on the current platform. A null implementation has been enabled (will not actually output gamepad events).")

# Optional import arm_kinematics_viz: only for availability self-check (does not affect the main flow)
try:
    import arm_kinematics_viz as _akv
    try:
        _akv.ArmKinematicsVisualizer().update_and_render(points=None, now=time.time())
        _AKV_AVAILABLE = True
        print("[Info] arm_kinematics_viz module is available (self-check passed).")
    except Exception as e:
        _AKV_AVAILABLE = False
        print(f"[Warning] arm_kinematics_viz was imported successfully but self-check failed: {e}")
except Exception as e:
    _AKV_AVAILABLE = False
    print(f"[Warning] Failed to import arm_kinematics_viz: {e}")


def start_arm_viz(cam: int = 0,
                  width: int = 640,
                  height: int = 480,
                  flip: bool = True,
                  history_sec: float = 6.0,
                  panel_size=(800, 540),
                  window_name: str = 'Arm Kinematics Viz (press q to quit)') -> int:
    """
    Starts the "Arm Kinematics Visualization" with minimal coupling in main.py.
    It is only imported and run when explicitly called, without affecting the current main flow.
    Return value: 0 for normal exit; non-0 for initialization failure (e.g., camera cannot be opened).
    """
    from run_arm_viz import run_arm_viz as _run_arm_viz
    return _run_arm_viz(cam=cam, width=width, height=height, flip=flip,
                        history_sec=history_sec, panel_size=panel_size,
                        window_name=window_name)

# ================== Parameters ==================
H_FLIP               = True
RATE_LIMIT_HZ        = 240
STICK_MAX            = 32000

# -- Extra HUD: Arm Kinematics Visualization (second window) --
AKV_HUD              = True           # Display a second window simultaneously (based on arm_kinematics_viz)
AKV_HISTORY_SEC      = 6.0
AKV_PANEL_SIZE       = (800, 540)
AKV_WINDOW_NAME      = 'Arm Kinematics Viz (press q to quit)'

# -- Right Hand: Real-time control (velocity -> joystick; gated by "hand gesture switch") --
RIGHT_HISTORY_SEC    = 0.20
RIGHT_SMOOTH_ALPHA   = 0.8
RIGHT_VEL_THRESH     = 0.0008
RIGHT_DEADZONE_SPD   = 0.0008
INVERT_X             = False
INVERT_Y             = True

# -- Record -> Burst Replay (triggered by raising the left hand) --
HAND_UP_MARGIN       = 0.03          # Wrist y < Shoulder y - margin determines hand is raised
RECORD_SEC           = 8.0           # ★ 8 seconds
REPLAY_COMPRESS      = 0.15
REPLAY_GAIN          = 1.8
DEADZONE_RADIUS      = 0.0015

# -- Hands Label Related -- 
SWAP_POSE_HANDS      = False         # Swap left/right for Pose skeleton (used for left hand raise detection)
SWAP_HANDS_LABEL     = False         # Swap the 'Left'/'Right' labels from Hands entirely (no correction)
GESTURE_HAND_LABEL   = 'Left'        # ★ Which hand's "open/fist" gesture to use as a switch: 'Left' or 'Right'

# -- Hand Gesture Detection Thresholds (with hysteresis for stability) --
OPEN_THRESH          = 1.6           # Open threshold (recommended 1.7~2.2)
CLOSE_THRESH         = 1.2           # Close threshold (must be < OPEN_THRESH)

HUD                  = True

# ================== Initialization ==================
mp_pose   = mp.solutions.pose
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_style  = mp.solutions.drawing_styles
gamepad   = _VX360Gamepad()

cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)

# Arm Kinematics Visualizer (second HUD) instance
akv_viz = None
if AKV_HUD and _AKV_AVAILABLE:
    try:
        akv_viz = _akv.ArmKinematicsVisualizer(history_sec=AKV_HISTORY_SEC, panel_size=AKV_PANEL_SIZE)
        print(f"[Info] Second window enabled: {AKV_WINDOW_NAME} size={AKV_PANEL_SIZE}, history={AKV_HISTORY_SEC}s")
    except Exception as e:
        akv_viz = None
        print(f"[Warning] Failed to create ArmKinematicsVisualizer: {e}")

# Right hand: history for real-time control and EMA
r_hist = deque()
rvx_ema = 0.0
rvy_ema = 0.0

# Recording state machine
state = "IDLE"                      # IDLE / RECORD / REPLAY
record_buf = []                     # Stores the right hand trajectory [(t,x,y)]
rec_t0 = None
next_tick = time.perf_counter()

# Hand gesture hysteresis state & debugging
gesture_open_state = False
last_openness = None

# ================== Utility Functions ==================
def clamp(v, lo, hi): return max(lo, min(hi, v))

def vel_to_stick(vx, vy):
    spd = math.hypot(vx, vy)
    if spd < RIGHT_DEADZONE_SPD: return 0, 0
    ux, uy = vx/(spd+1e-9), vy/(spd+1e-9)
    mag = STICK_MAX * math.tanh(200.0 * spd)
    sx, sy = ux*mag, uy*mag
    if INVERT_X: sx = -sx
    if INVERT_Y: sy = -sy
    return int(clamp(sx, -STICK_MAX, STICK_MAX)), int(clamp(sy, -STICK_MAX, STICK_MAX))

def estimate_velocity(hist_list):
    n = len(hist_list)
    tN, xN, yN = hist_list[-1]
    t0, x0, y0 = hist_list[0]
    dt = max(tN - t0, 1e-6)
    if n >= 4:
        ts = np.array([p[0] for p in hist_list], dtype=np.float64)
        xs = np.array([p[1] for p in hist_list], dtype=np.float64)
        ys = np.array([p[2] for p in hist_list], dtype=np.float64)
        A = np.vstack([ts - ts[0], np.ones_like(ts)]).T
        ax, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        ay, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        return float(ax), float(ay)
    else:
        return (xN - x0)/dt, (yN - y0)/dt

def simplify_path(points, eps=0.0008):
    if len(points) < 3: return points
    P = np.array([(p[1], p[2]) for p in points], dtype=np.float64)
    def rdp_idx(Q, eps):
        if len(Q) < 3: return [0, len(Q)-1]
        start, end = Q[0], Q[-1]
        se = end - start; L = np.linalg.norm(se) + 1e-12
        d = np.abs(np.cross(Q[1:-1] - start, se) / L)
        i = int(np.argmax(d)); dmax = d[i]
        if dmax > eps:
            k = i + 1
            Lidx = rdp_idx(Q[:k+1], eps)
            Ridx = rdp_idx(Q[k:],   eps)
            return Lidx[:-1] + [j+k for j in Ridx]
        else:
            return [0, len(Q)-1]
    keep = sorted(set(rdp_idx(P, eps)))
    return [points[i] for i in keep]

def pos_to_stick(dx, dy, gain=1.0):
    sx = STICK_MAX * math.tanh(220.0 * dx * gain)
    sy = STICK_MAX * math.tanh(220.0 * dy * gain)
    if INVERT_X: sx = -sx
    if INVERT_Y: sy = -sy
    return int(clamp(sx, -STICK_MAX, STICK_MAX)), int(clamp(sy, -STICK_MAX, STICK_MAX))

def replay_trajectory(buf, compress=0.15, gain=1.8):
    if len(buf) < 2: return
    buf = simplify_path(buf, eps=0.0008)
    t0 = buf[0][0]; T = buf[-1][0] - t0
    if T <= 0: return
    norm = [((t - t0)/T, x, y) for (t,x,y) in buf]
    last_emit = time.perf_counter(); min_dt = 1.0 / RATE_LIMIT_HZ
    steps = max(60, int(600 * compress + 1))
    for i in range(steps+1):
        frac = i / steps
        lo, hi = 0, len(norm)-1
        while lo < hi:
            mid = (lo+hi)//2
            if norm[mid][0] < frac: lo = mid+1
            else: hi = mid
        idx = lo
        if idx == 0: _, x, y = norm[0]
        elif idx >= len(norm): _, x, y = norm[-1]
        else:
            t1,x1,y1 = norm[idx-1]; t2,x2,y2 = norm[idx]
            r = 0 if t2-t1<1e-9 else (frac - t1)/(t2 - t1)
            x = x1 + r*(x2 - x1); y = y1 + r*(y2 - y1)
        dx, dy = x - norm[0][1], y - norm[0][2]
        if abs(dx)+abs(dy) < DEADZONE_RADIUS: sx, sy = 0, 0
        else: sx, sy = pos_to_stick(dx, dy, gain=gain)
        now = time.perf_counter()
        if now - last_emit < min_dt: time.sleep(min_dt - (now - last_emit))
        gamepad.left_joystick(x_value=sx, y_value=sy); gamepad.update()
        last_emit = time.perf_counter()
    for _ in range(18):
        gamepad.left_joystick(x_value=0, y_value=0); gamepad.update(); time.sleep(0.005)

# ---- Get Hands Label ----
def get_handedness_label(h):
    lbl = h.classification[0].label  # 'Left' or 'Right'
    if SWAP_HANDS_LABEL:
        return 'Right' if lbl == 'Left' else 'Left'
    return lbl

# ---- "Open/Fist" detection for a specific hand (Left/Right) (based on label + hysteresis) ----
def hand_is_open(hands_res, img_w, img_h, target_label='Right'):
    """
    Selects the hand with target_label using Hands' Left/Right label, 
    calculates openness, and determines the state using a hysteresis threshold.
    Returns: True / False / None
    """
    global gesture_open_state, last_openness
    if hands_res is None or not hands_res.multi_hand_landmarks:
        last_openness = None
        return None

    idx = None
    for i, handed in enumerate(hands_res.multi_handedness):
        if get_handedness_label(handed) == target_label:
            idx = i; break
    if idx is None:
        last_openness = None
        return None

    lms = hands_res.multi_hand_landmarks[idx].landmark

    def to_px(lm): return np.array([lm.x * img_w, lm.y * img_h], dtype=np.float64)

    wrist = to_px(lms[0])
    mcp9  = to_px(lms[9])
    tips  = [to_px(lms[i]) for i in (4,8,12,16,20)]

    palm_scale = np.linalg.norm(mcp9 - wrist) + 1e-6
    openness   = float(np.mean([np.linalg.norm(tp - wrist) for tp in tips]) / palm_scale)
    last_openness = openness

    # Hysteresis logic
    if gesture_open_state:
        if openness < CLOSE_THRESH:
            gesture_open_state = False
    else:
        if openness > OPEN_THRESH:
            gesture_open_state = True

    return gesture_open_state

# ================== Main Loop ==================
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while True:
        ok, frame = cap.read()
        if not ok: break
        if H_FLIP: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_pose  = pose.process(rgb)
        res_hands = hands.process(rgb)

        have = res_pose.pose_landmarks is not None
        rw_x = rw_y = lw_x = lw_y = None
        l_hand_up = False

        if have:
            lm = res_pose.pose_landmarks.landmark
            if SWAP_POSE_HANDS:
                R_WRIST, R_SHOUL = mp_pose.PoseLandmark.LEFT_WRIST,  mp_pose.PoseLandmark.LEFT_SHOULDER # mp_pose object holds the information on 
                L_WRIST, L_SHOUL = mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER
            else:
                R_WRIST, R_SHOUL = mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER
                L_WRIST, L_SHOUL = mp_pose.PoseLandmark.LEFT_WRIST,  mp_pose.PoseLandmark.LEFT_SHOULDER

            RW, RS = lm[R_WRIST], lm[R_SHOUL]
            LW, LS = lm[L_WRIST], lm[L_SHOUL]

            if getattr(RW,"visibility",1.0)>0.5 and getattr(RS,"visibility",1.0)>0.5:
                rw_x, rw_y = float(RW.x), float(RW.y)
            if getattr(LW,"visibility",1.0)>0.5 and getattr(LS,"visibility",1.0)>0.5:
                lw_x, lw_y = float(LW.x), float(LW.y)
                l_hand_up  = (lw_y < (float(LS.y) - HAND_UP_MARGIN))

            if HUD:
                mp_draw.draw_landmarks(frame, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
                cv2.circle(frame, (int(RW.x*w), int(RW.y*h)), 6, (255,  0,255), -1)
                cv2.circle(frame, (int(RS.x*w), int(RS.y*h)), 6, (  0,255,  0), -1)
                cv2.circle(frame, (int(LW.x*w), int(LW.y*h)), 6, (  0,255,255), -1)
                cv2.circle(frame, (int(LS.x*w), int(LS.y*h)), 6, (255,255,  0), -1)

        # "Open/Fist" gesture comes from GESTURE_HAND_LABEL (defaults to using the left hand as the switch)
        hand_open = hand_is_open(res_hands, w, h, target_label=GESTURE_HAND_LABEL)  # True/False/None

        now_t = time.time()

        # ===== Left Hand: Raising hand triggers recording of right hand trajectory (no output during recording) =====
        if state == "IDLE":
            if l_hand_up and (rw_x is not None):
                state = "RECORD"; record_buf = [(now_t, rw_x, rw_y)]; rec_t0 = now_t

        elif state == "RECORD":
            if rw_x is not None:
                record_buf.append((now_t, rw_x, rw_y))
            else:
                if record_buf:
                    lt, lx, ly = record_buf[-1]
                    record_buf.append((now_t, lx, ly))
            if (now_t - rec_t0) >= RECORD_SEC:
                state = "REPLAY"

        elif state == "REPLAY":
            replay_trajectory(record_buf, compress=REPLAY_COMPRESS, gain=REPLAY_GAIN)
            state = "IDLE"; record_buf = []; rec_t0 = None

        # ===== Right Hand: Real-time control (outputs only when not recording/replaying, and the selected hand is open) =====
        if state == "IDLE" and rw_x is not None and hand_open is not None and hand_open:
            r_hist.append((now_t, rw_x, rw_y))
            while r_hist and (now_t - r_hist[0][0]) > RIGHT_HISTORY_SEC:
                r_hist.popleft()
            if len(r_hist) >= 2:
                rvx, rvy = estimate_velocity(r_hist)
                rvx_ema = RIGHT_SMOOTH_ALPHA*rvx + (1-RIGHT_SMOOTH_ALPHA)*rvx_ema
                rvy_ema = RIGHT_SMOOTH_ALPHA*rvy + (1-RIGHT_SMOOTH_ALPHA)*rvy_ema
                spd = math.hypot(rvx_ema, rvy_ema)
                now_tick = time.perf_counter()
                if now_tick >= next_tick:
                    next_tick = now_tick + (1.0 / RATE_LIMIT_HZ)
                    if spd > RIGHT_VEL_THRESH:
                        sx, sy = vel_to_stick(rvx_ema, rvy_ema)
                    else:
                        sx, sy = 0, 0
                    gamepad.left_joystick(x_value=sx, y_value=sy); gamepad.update()
        else:
            # Condition for output not met or during recording/replaying: can add smooth return-to-zero logic here
            pass

        # ===== HUD =====
        if HUD:
            open_txt = "?" if hand_open is None else ("Y" if hand_open else "n")
            s = f"STATE:{state} | L-up:{'Y' if l_hand_up else 'n'} | {GESTURE_HAND_LABEL}-open:{open_txt}"
            if last_openness is not None:
                s += f" | open={last_openness:.2f} (>{OPEN_THRESH:.1f}/{CLOSE_THRESH:.1f})"
            cv2.putText(frame, s, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0,255,0) if state!='IDLE' else (0,0,255), 2)
            if state == "RECORD" and rec_t0 is not None:
                cv2.putText(frame, f"Recording RIGHT {now_t-rec_t0:.1f}/{RECORD_SEC:.1f}s pts:{len(record_buf)}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,0), 2)
            cv2.putText(frame, f"Right=open -> control (gated by {GESTURE_HAND_LABEL}) | Left up -> record 8s -> replay",
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        if akv_viz is not None:
            # Build points for ArmKinematicsVisualizer from current MediaPipe Pose result
            try:
                akv_points = _akv.extract_points_from_mediapipe(res_pose)
            except Exception:
                akv_points = None
            lw_pt = (lw_x, lw_y) if (lw_x is not None and lw_y is not None) else None
            rw_pt = (rw_x, rw_y) if (rw_x is not None and rw_y is not None) else None
            panel = akv_viz.update_and_render(points=akv_points, now=now_t, lw=lw_pt, rw=rw_pt)
            cv2.imshow(AKV_WINDOW_NAME, panel)
        cv2.imshow("Control gated by other hand open/close (q to quit)", frame)

cap.release()
cv2.destroyAllWindows()