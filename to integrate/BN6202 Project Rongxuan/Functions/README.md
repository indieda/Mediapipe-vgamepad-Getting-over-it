# Mediapipe-vgamepad-Getting-over-it
# 🎮 Gesture-Controlled Gamepad (Mediapipe + vgamepad)

This project allows you to control a virtual Xbox360 gamepad using **hand gestures and body tracking** via [Mediapipe](https://github.com/google/mediapipe).  
It supports **real-time joystick control with your right hand** and **record + replay mode triggered by left hand gestures**.

---

## ✨ Features
- 🖐️ **Hand gesture switch**: Open/close hand to enable or disable control.
- 🤚 **Left hand up**: Record right hand motion for 8s, then auto replay.
- 🎮 **Virtual joystick**: Maps motion to Xbox360 joystick using [vgamepad](https://pypi.org/project/vgamepad/).
- 📊 **HUD Overlay**: Displays state, gesture openness, and recording info.

---

## 📦 Installation

# 1. Create env
conda create -n gesture-pad python=3.10
conda activate gesture-pad

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Windows only) Install ViGEmBus Driver
- Search **"ViGEmBus Driver Nefarius"**
- Download and install the official package (requires admin privileges)
- Restart Windows after installation
