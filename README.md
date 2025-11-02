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



# 2nd nov 2025:

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

## 📺 Real-time Monitoring UI

The new Streamlit monitoring console offers a side-by-side view of live video
and telemetry:

- Left panel (40% width): four bar charts (Left/Right Elbow & Shoulder angles) and two line charts refreshed in real time (Left/Right wrist speed plus elbow angle trends).
- Right panel (60% width): a 640×480 OpenCV video stream shared with `main.py`.
- Data is exchanged through a multiprocessing-safe `DataBridge`, ensuring the
	UI can run in a separate process.

### Launch everything together

The `ui_launcher.py` helper spins up the shared bridge, the Streamlit UI, and
the original `main.py` loop in one command.

```bash
python ui_launcher.py --streamlit-port 8501
```

Options:

- `--no-main` Launch only the dashboard with synthetic demo data.
- `--keep-mock` Keep synthetic demo data running even after `main.py` connects.
- `--video-source 0` Select the OpenCV capture source for mock video frames.

Once running, open your browser at <http://localhost:8501>.

### Standalone usage

To connect the dashboard to a custom producer, set these environment variables
before launching Streamlit:

- `DATA_BRIDGE_HOST`
- `DATA_BRIDGE_PORT`
- `DATA_BRIDGE_AUTHKEY` (hex string)

Then run:

```bash
streamlit run streamlit_ui.py
```

The `DataBridge` API exposes:

```python
DataBridge.get_video_frame() -> np.ndarray
DataBridge.get_bar_chart_data() -> List[dict]
DataBridge.get_line_chart_data() -> List[dict]
DataBridge.update_data(data_type, payload)
```

Refer to `data_bridge.py` for detailed documentation.

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
