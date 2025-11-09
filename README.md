# GestureCam

A gesture-controlled camera system built with Python, OpenCV, and MediaPipe Hand Landmarker. Control your camera using hand gestures in real-time - take photos, adjust zoom, and manage capture modes all without touching your device.

## Features

- **Real-time Hand Gesture Recognition** using MediaPipe Hand Landmarker
- **Multiple Camera Control Gestures**:
  - Open palm → Enable Gesture Control Mode
  - V Sign (Peace) → Take photo (3-second countdown)
  - Index finger only → Zoom in
  - Pinky finger only → Zoom out
  - Three-finger salute → Disable Gesture Control Mode
  - Thumbs up → Save captured photo
  - Thumbs down → Discard captured photo
  - OK sign (index + thumb circle) → Enter Edit Mode
- **State Machine Architecture** with multiple modes:
  - Preview Mode
  - Gesture Control Mode
  - Capture Countdown Mode
  - Review Mode
  - **Edit Mode** - Real-time photo editing with gesture controls
- **Photo Editing Features**:
  - Adjust brightness, contrast, saturation, warmth, and sharpness
  - Wrist-based parameter selection (hover 2 seconds to select)
  - Fist gesture for slider control
  - Two-hand gestures for save/discard (two open palms = save, two fists = discard)
  - Real-time preview of edits
- **Multi-frame Stability Detection** for reliable gesture recognition
- **Digital Zoom** with multiple zoom levels
- **Photo Capture** with countdown timer and review system
- **Visual Feedback** with on-screen overlays, progress indicators, and gesture hints for all modes
- **Fixed Display Window** - Always 1280x720 regardless of camera resolution

## Requirements

- Python 3.8+
- Webcam/Camera
- MediaPipe
- OpenCV
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Avijit-D/GestureCam.git
cd GestureCam
```

2. Create and activate a virtual environment:
```bash
python -m venv cv
# On Windows:
.\cv\Scripts\Activate.ps1
# On Linux/Mac:
source cv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. The MediaPipe hand landmarker model will be automatically downloaded on first run (saved to `models/hand_landmarker.task`).

## Usage

Run the application:
```bash
python main.py
```

### Gesture Controls

#### Mode Switching
- **Open Palm (Hold 0.4s)**: Enter Gesture Control Mode from Preview Mode
- **Three-Finger Salute**: Exit Gesture Control Mode (returns to Preview)

#### Camera Controls (in Gesture Mode)
- **V Sign (Peace)**: Start 3-second countdown, then capture photo
- **Index Finger Only**: Zoom in (cycles through 1.0x, 1.5x, 2.0x)
- **Pinky Finger Only**: Zoom out (cycles through zoom levels)

#### Photo Review (after capture)
- **Thumbs Up**: Save the captured photo
- **Thumbs Down**: Discard the captured photo
- **Open Palm**: Enter Edit Mode
- **Rock Sign**: Exit Gesture Mode and return to Preview

#### Photo Editing (in Edit Mode)
- **Wrist Position**: Hover over parameter boxes for 2 seconds to select
- **Fist Gesture**: Move wrist left/right to adjust selected parameter value
- **Two Open Palms**: Save edited photo
- **Two Fists**: Discard edits and return to Gesture Mode
- **Available Parameters**: Brightness, Contrast, Saturation, Warmth, Sharpness

### Gesture Detection Requirements

- **Open Palm**: Requires 3.8+ extended fingers, 0.6+ confidence, stable hand position
- **Index/Pinky Only**: Only the specified finger extended, all others folded
- **Multi-frame validation**: Gestures must be detected consistently across multiple frames

## Project Structure

```
GestureCam/
├── main.py                 # Main application entry point
├── gesture_detector.py     # Hand gesture recognition using MediaPipe
├── camera_controller.py    # Camera actions and photo management
├── utils/
│   └── drawing.py         # Visualization utilities for landmarks and overlays
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── models/                # MediaPipe model files (auto-downloaded)
└── captures/              # Saved photos directory
```

## How It Works

1. **Hand Detection**: MediaPipe Hand Landmarker detects up to 2 hands and extracts 21 landmarks per hand
2. **Gesture Classification**: Custom algorithms analyze finger positions and movements to classify gestures
3. **State Management**: A state machine manages mode transitions and gesture permissions
4. **Action Execution**: Detected gestures trigger camera actions (zoom, capture, mode switching)
5. **Visual Feedback**: Real-time overlays show detected gestures, mode status, and countdown timers

## Technical Details

### Gesture Detection Algorithm

- Uses MediaPipe's Hand Landmarker for robust hand tracking
- Implements multi-frame stability checks to prevent false positives
- Validates finger extension using joint angles and positions
- Tracks hand movement over 8 frames for stability validation

### State Machine Modes

- **Preview Mode**: Default mode, thumbs up gesture to enter Gesture Mode
- **Gesture Control Mode**: Full gesture control active (zoom, capture)
- **Capture Countdown Mode**: 3-second countdown, gestures disabled
- **Review Mode**: Display captured photo, review gestures enabled
- **Edit Mode**: Real-time photo editing with parameter adjustment controls

### Zoom Implementation

- Digital zoom using center crop and resize
- Three zoom levels: 1.0x, 1.5x, 2.0x
- Smooth transitions between zoom levels

## Configuration

Key parameters can be adjusted in `main.py`:

- `hold_duration_required`: Time to hold thumbs up (default: 0.3s)
- `frames_required_to_confirm`: Frames needed to confirm gesture (default: 4)
- `stability_threshold_norm`: Maximum hand movement for stability (default: 0.12)
- `countdown_duration`: Photo countdown timer (default: 3.0s)
- `DISPLAY_WIDTH` / `DISPLAY_HEIGHT`: Fixed window size (default: 1280x720)
- `DETECTION_WIDTH`: Detection frame width for performance (default: 480)

## Troubleshooting

- **Camera not opening**: Check camera permissions and ensure no other application is using the camera
- **Gestures not detected**: Ensure good lighting and hand visibility in frame
- **Model download fails**: Check internet connection; model will be downloaded on first run
- **Performance issues**: Lower camera resolution or reduce frame processing rate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and personal use.

## Acknowledgments

- Built with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google
- Uses [OpenCV](https://opencv.org/) for computer vision operations

## Author

Avijit-D
