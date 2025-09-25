#!/usr/bin/env python3
"""
Standalone runner for ArmKinematicsVisualizer (no vgamepad required).

Usage examples:
  python run_arm_viz.py                 # default camera 0, 640x480, flip on
  python run_arm_viz.py --cam 1 --no-flip --width 1280 --height 720

Press 'q' in the window to quit.
"""
from __future__ import annotations
import argparse
import time
import sys
from typing import Tuple

import cv2

try:
    import mediapipe as mp
    _HAVE_MP = True
except Exception as e:
    mp = None
    _HAVE_MP = False
    _MP_ERR = e

import arm_kinematics_viz as akv


def parse_args():
    p = argparse.ArgumentParser(description='Run Arm Kinematics Visualization (webcam + MediaPipe Pose).')
    p.add_argument('--cam', type=int, default=0, help='Camera index (default: 0)')
    p.add_argument('--width', type=int, default=640, help='Capture width (default: 640)')
    p.add_argument('--height', type=int, default=480, help='Capture height (default: 480)')
    p.add_argument('--flip', dest='flip', action='store_true', help='Mirror the camera image (default)')
    p.add_argument('--no-flip', dest='flip', action='store_false', help='Do not mirror the camera image')
    p.set_defaults(flip=True)
    return p.parse_args()


def run_arm_viz(cam: int = 0,
                width: int = 640,
                height: int = 480,
                flip: bool = True,
                history_sec: float = 6.0,
                panel_size: Tuple[int, int] = (800, 540),
                window_name: str = 'Arm Kinematics Viz (press q to quit)') -> int:
    """Run the ArmKinematicsVisualizer loop.

    Parameters are intentionally simple to keep coupling low when called from other modules.
    Returns process exit code (0 success, non-zero on basic init error).
    """
    # Initialize visualizer panel (rendered independently of the camera frame)
    viz = akv.ArmKinematicsVisualizer(history_sec=history_sec, panel_size=panel_size)

    # The path to your video file
    video_path = 'path/to/your/video.mp4' 
    cap = cv2.VideoCapture(video_path)

    # cap = cv2.VideoCapture(0)
    # cap.set(3, 640); cap.set(4, 480)

    # You NO LONGER need cap.set() for a pre-recorded video.
    # The video's resolution is already fixed. These lines will have no effect.
######################################################################################
# Uncomment this section to use live webcam.
    ## Camera
    #cap = cv2.VideoCapture(cam)
    # if not cap.isOpened():
    #     print(f"[ERROR] Cannot open camera index {cam}.")
    #     return 1
    # if width:
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # if height:
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
######################################################################################

    # MediaPipe Pose (optional fallback if unavailable)
    if _HAVE_MP:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False,
                            model_complexity=1,
                            enable_segmentation=False,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
    else:
        pose = None
        print(f"[WARN] mediapipe not available: {_MP_ERR}\n       Running in panel-only mode (no angles/speeds).")

    last_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[WARN] Failed to read from camera. Exiting.')
                break
            if flip:
                frame = cv2.flip(frame, 1)

            now = time.time()

            points = None
            lw = rw = None
            if pose is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_pose = pose.process(rgb)
                points = akv.extract_points_from_mediapipe(res_pose)
                if points:
                    lw = points.get('L_WRIST')
                    rw = points.get('R_WRIST')

            panel = viz.update_and_render(points=points, now=now, lw=lw, rw=rw)

            # Show FPS on panel
            now_perf = time.perf_counter()
            dt = now_perf - last_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_time = now_perf
            cv2.putText(panel, f"FPS: {fps:5.1f}", (10, panel.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        if pose is not None:
            pose.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main():
    args = parse_args()
    return run_arm_viz(cam=args.cam, width=args.width, height=args.height, flip=args.flip)


if __name__ == '__main__':
    sys.exit(main())
