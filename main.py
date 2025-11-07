"""Entry point for gesture-driven camera controller.

Usage:
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

import enum
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from camera_controller import CameraController
from gesture_detector import GestureRecognition
from utils.drawing import (
    draw_countdown,
    draw_hold_progress,
    draw_landmarks,
    draw_mode_banner,
    draw_review_prompts,
    overlay_gestures,
)


class Mode(enum.Enum):
    PREVIEW = "Preview"
    GESTURE = "Gesture Mode"
    AUTO_FRAMING = "Auto-Framing Active"
    COUNTDOWN = "Capture Countdown"
    REVIEW = "Review"


MODE_ALLOWED_GESTURES: dict[Mode, Set[str]] = {
    Mode.PREVIEW: {"open_palm"},
    Mode.GESTURE: {"two_finger_v", "three_finger_salute", "index_only", "pinky_only"},
    Mode.AUTO_FRAMING: {"open_palm"},
    Mode.COUNTDOWN: set(),
    Mode.REVIEW: {"thumbs_up", "thumbs_down", "open_palm", "three_finger_salute"},
}


GESTURE_COOLDOWNS = {
    "open_palm": 0.8,
    "two_finger_v": 1.0,
    "three_finger_salute": 1.0,
    "index_only": 0.5,
    "pinky_only": 0.5,
    "thumbs_up": 1.0,
    "thumbs_down": 1.0,
}


def apply_digital_zoom(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Crop from the center and resize back to simulate digital zoom."""

    if zoom_factor <= 1.0:
        return frame

    height, width = frame.shape[:2]
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    if new_width <= 0 or new_height <= 0:
        return frame

    start_x = max((width - new_width) // 2, 0)
    start_y = max((height - new_height) // 2, 0)
    end_x = start_x + new_width
    end_y = start_y + new_height

    cropped = frame[start_y:end_y, start_x:end_x]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def prettify_labels(labels: Iterable[str]) -> List[str]:
    return [label.replace("_", " ").title() for label in labels]


def compute_auto_framing_suggestions(
    hand_landmarks: Sequence[Sequence[float]],
) -> List[str]:
    """Derive simple framing suggestions based on wrist position."""

    if not hand_landmarks:
        return ["Hold position"]

    # Use first detected hand wrist as reference.
    wrist = hand_landmarks[0][0]
    x, y = wrist[0], wrist[1]

    suggestions: List[str] = []
    if x < 0.4:
        suggestions.append("Move Right")
    elif x > 0.6:
        suggestions.append("Move Left")

    if y < 0.4:
        suggestions.append("Move Down")
    elif y > 0.6:
        suggestions.append("Move Up")

    if not suggestions:
        suggestions.append("Great framing")

    return suggestions


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    controller = CameraController()
    detector = GestureRecognition()

    current_mode = Mode.PREVIEW
    countdown_start: float | None = None
    countdown_duration = 3.0
    last_gesture_trigger: dict[str, float] = {}
    review_frame: np.ndarray | None = None
    status_messages: List[str] = []
    photo_saved_timestamp: Optional[float] = None  # Timestamp when photo was saved
    photo_saved_message_duration = 2.0  # Show message for 2 seconds
    cooldown_until: float = 0.0
    hold_start_time: Optional[float] = None
    holding_hand_id: Optional[str] = None
    hold_progress: float = 0.0
    
    # Multi-frame stability tracking
    open_palm_valid_frames: Dict[str, int] = {}  # hand_id -> consecutive valid frames
    open_palm_invalid_frames: Dict[str, int] = {}  # hand_id -> consecutive invalid frames
    wrist_history: Dict[str, deque] = {}  # hand_id -> deque of (timestamp, x, y) over last 8 frames
    frame_height_px: int = 480  # Will be updated from actual frame
    
    hold_duration_required = 0.4
    frames_required_to_confirm = 2
    frames_required_to_reset = 3
    stability_window_frames = 8
    stability_threshold_px = 30

    def reset_hold_state() -> None:
        nonlocal hold_start_time, holding_hand_id, hold_progress, status_messages
        hold_start_time = None
        holding_hand_id = None
        hold_progress = 0.0
        if status_messages == ["Hold steady to switch"]:
            status_messages = []

    def is_hand_stable_multi_frame(hand_id: str, wrist_x: float, wrist_y: float, timestamp: float, frame_width: int, frame_height: int) -> bool:
        """Check if hand is stable over last 8 frames (average movement <= 35px)."""
        if hand_id not in wrist_history:
            wrist_history[hand_id] = deque(maxlen=stability_window_frames)
        
        history = wrist_history[hand_id]
        history.append((timestamp, wrist_x, wrist_y))
        
        if len(history) < 2:
            return True  # Need at least 2 frames to compute movement
        
        # Calculate average movement over the window
        total_movement = 0.0
        for i in range(1, len(history)):
            _, x1, y1 = history[i-1]
            _, x2, y2 = history[i]
            # Convert normalized to pixels
            dx_px = abs(x2 - x1) * frame_width
            dy_px = abs(y2 - y1) * frame_height
            movement_px = (dx_px**2 + dy_px**2)**0.5
            total_movement += movement_px
        
        avg_movement = total_movement / (len(history) - 1)
        return avg_movement <= stability_threshold_px

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp = time.time()
            timestamp_ms = int(timestamp * 1000)
            cooldown_active = timestamp < cooldown_until
            
            frame_height_px, frame_width_px = frame.shape[:2]

            gestures = detector.classify(frame, timestamp_ms)
            gesture_names = sorted({gesture.name for gesture in gestures})
            allowed = MODE_ALLOWED_GESTURES[current_mode]

            def gesture_ready(name: str) -> bool:
                cooldown = GESTURE_COOLDOWNS.get(name, 0.3)
                last_ts = last_gesture_trigger.get(name)
                return last_ts is None or (timestamp - last_ts) >= cooldown

            active_gestures = [
                g for g in gesture_names if g in allowed and gesture_ready(g)
            ]

            # Multi-frame open palm detection with stability (only for turning ON Gesture Mode)
            open_palm_candidate: Optional[Tuple[float, str, Sequence[float], float]] = None
            if current_mode == Mode.PREVIEW and not cooldown_active:
                for gesture in gestures:
                    if gesture.name != "open_palm" or gesture.hand_id is None:
                        continue
                    
                    hand_id = gesture.hand_id
                    confidence = gesture.confidence
                    extended = gesture.metadata.get("extended_count", 0.0)
                    wrist = gesture.metadata.get("wrist")
                    palm_radius = gesture.metadata.get("palm_radius", 0.0)
                    
                    if not wrist or len(wrist) < 2:
                        continue
                    
                    # Check palm size: palm_radius >= 0.10 * frame_height (in normalized units)
                    palm_radius_px = palm_radius * frame_height_px
                    min_palm_size_px = 0.10 * frame_height_px
                    if palm_radius_px < min_palm_size_px:
                        # Invalid frame
                        open_palm_invalid_frames[hand_id] = open_palm_invalid_frames.get(hand_id, 0) + 1
                        open_palm_valid_frames[hand_id] = 0
                        continue
                    
                    # Check multi-frame stability
                    wrist_x, wrist_y = wrist[0], wrist[1]
                    if not is_hand_stable_multi_frame(hand_id, wrist_x, wrist_y, timestamp, frame_width_px, frame_height_px):
                        # Invalid frame
                        open_palm_invalid_frames[hand_id] = open_palm_invalid_frames.get(hand_id, 0) + 1
                        open_palm_valid_frames[hand_id] = 0
                        continue
                    
                    # Valid frame - increment counter
                    open_palm_valid_frames[hand_id] = open_palm_valid_frames.get(hand_id, 0) + 1
                    open_palm_invalid_frames[hand_id] = 0
                    
                    # Require 6 consecutive valid frames to confirm
                    if open_palm_valid_frames[hand_id] >= frames_required_to_confirm:
                        open_palm_candidate = (confidence, hand_id, wrist, palm_radius)
                        break
                
                # Reset counters for hands that didn't appear this frame
                for hand_id in list(open_palm_valid_frames.keys()):
                    if not any(g.hand_id == hand_id for g in gestures if g.name == "open_palm"):
                        open_palm_invalid_frames[hand_id] = open_palm_invalid_frames.get(hand_id, 0) + 1
                        open_palm_valid_frames[hand_id] = 0
                        
                        # Reset after 5 consecutive invalid frames
                        if open_palm_invalid_frames[hand_id] >= frames_required_to_reset:
                            open_palm_valid_frames[hand_id] = 0
                            open_palm_invalid_frames[hand_id] = 0
                            if holding_hand_id == hand_id:
                                reset_hold_state()

            holding_active = False
            if current_mode == Mode.PREVIEW:
                if cooldown_active:
                    reset_hold_state()
                elif open_palm_candidate:
                    _, candidate_id, _, _ = open_palm_candidate
                    if holding_hand_id and holding_hand_id != candidate_id:
                        reset_hold_state()
                    if hold_start_time is None:
                        hold_start_time = timestamp
                        holding_hand_id = candidate_id
                    hold_progress = min(
                        (timestamp - hold_start_time) / hold_duration_required, 1.0
                    )
                    holding_active = True
                    status_messages = ["Hold steady to switch"]
                    if hold_progress >= 1.0:
                        reset_hold_state()
                        cooldown_until = timestamp + 1.5
                        current_mode = Mode.GESTURE
                        status_messages = ["Gesture Control Mode enabled"]
                        print(f"[ACTION] Gesture Control Mode enabled")
                        review_frame = None
                        countdown_start = None
                        active_gestures = []
                        # Reset multi-frame counters after successful mode switch
                        if candidate_id in open_palm_valid_frames:
                            open_palm_valid_frames[candidate_id] = 0
                            open_palm_invalid_frames[candidate_id] = 0
                        continue
                else:
                    reset_hold_state()
            else:
                reset_hold_state()

            # Ignore gestures during countdown, hold, or cooldown
            if holding_active or cooldown_active or current_mode == Mode.COUNTDOWN:
                active_gestures = []

            log_messages: List[str] = []

            if current_mode == Mode.PREVIEW:
                pass

            elif current_mode == Mode.GESTURE:
                if "two_finger_v" in active_gestures:
                    current_mode = Mode.COUNTDOWN
                    countdown_start = timestamp
                    last_gesture_trigger["two_finger_v"] = timestamp
                    status_messages = ["Countdown started"]
                    log_messages.append("Photo countdown started")
                elif "three_finger_salute" in active_gestures:
                    current_mode = Mode.PREVIEW
                    last_gesture_trigger["three_finger_salute"] = timestamp
                    status_messages = ["Preview Mode"]
                    log_messages.append("Gesture Mode disabled")
                elif "index_only" in active_gestures:
                    last_gesture_trigger["index_only"] = timestamp
                    message = controller.zoom_in()
                    log_messages.append(message)
                    status_messages = [message]
                elif "pinky_only" in active_gestures:
                    last_gesture_trigger["pinky_only"] = timestamp
                    message = controller.zoom_out()
                    log_messages.append(message)
                    status_messages = [message]

            elif current_mode == Mode.AUTO_FRAMING:
                if "open_palm" in active_gestures:
                    current_mode = Mode.GESTURE
                    last_gesture_trigger["open_palm"] = timestamp
                    log_messages.append("Gesture Control Mode enabled")

            elif current_mode == Mode.REVIEW:
                if "thumbs_up" in active_gestures:
                    last_gesture_trigger["thumbs_up"] = timestamp
                    path = controller.save_pending_capture()
                    if path:
                        log_messages.append(f"Saved capture to {path.name}")
                        status_messages = [f"Saved {path.name}"]
                        photo_saved_timestamp = timestamp  # Track when photo was saved
                    else:
                        status_messages = ["No capture to save"]
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "thumbs_down" in active_gestures:
                    last_gesture_trigger["thumbs_down"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Capture discarded"]
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "open_palm" in active_gestures:
                    last_gesture_trigger["open_palm"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Return to gesture mode"]
                    current_mode = Mode.GESTURE
                    review_frame = None
                elif "three_finger_salute" in active_gestures:
                    last_gesture_trigger["three_finger_salute"] = timestamp
                    controller.discard_pending_capture()
                    status_messages = ["Gesture Mode disabled"]
                    current_mode = Mode.PREVIEW
                    review_frame = None
                    log_messages.append("Gesture Mode disabled from review")

            if current_mode == Mode.COUNTDOWN:
                if countdown_start is None:
                    countdown_start = timestamp
                elapsed = timestamp - countdown_start
                remaining = countdown_duration - elapsed
                if remaining <= 0:
                    countdown_start = None
                    # Stay in Gesture Mode but show review
                    zoomed_capture = apply_digital_zoom(frame, controller.get_zoom_level())
                    controller.store_capture(zoomed_capture, timestamp)
                    review_frame = controller.peek_pending_capture()
                    current_mode = Mode.REVIEW
                    status_messages = ["Capture ready for review"]
                    log_messages.append("Capture completed, entering review mode")
            
            # Auto-hide "Photo Saved" message after 2 seconds
            if photo_saved_timestamp is not None:
                elapsed_since_save = timestamp - photo_saved_timestamp
                if elapsed_since_save >= photo_saved_message_duration:
                    # Clear only the saved message if it's still showing
                    if status_messages:
                        status_messages = [
                            msg for msg in status_messages if "Saved" not in msg
                        ]
                    photo_saved_timestamp = None

            for message in log_messages:
                print(f"[ACTION] {message}")

            landmark_entries = detector.get_landmarks()
            landmarks = [coords.tolist() for _, coords in landmark_entries]

            if current_mode == Mode.REVIEW and review_frame is not None:
                display_frame = review_frame.copy()
            else:
                display_frame = frame.copy()
                display_frame = draw_landmarks(display_frame, landmarks)

            if current_mode == Mode.AUTO_FRAMING:
                suggestions = compute_auto_framing_suggestions(landmarks)
                display_frame = overlay_gestures(
                    display_frame,
                    suggestions,
                    origin=(10, 70),
                )

            if current_mode == Mode.COUNTDOWN and countdown_start is not None:
                elapsed = timestamp - countdown_start
                remaining = max(countdown_duration - elapsed, 0.0)
                display_frame = draw_countdown(display_frame, remaining)

            display_frame = draw_mode_banner(display_frame, current_mode.value)

            if holding_active:
                display_frame = draw_hold_progress(display_frame, hold_progress)

            if current_mode not in {Mode.COUNTDOWN, Mode.REVIEW} and not holding_active:
                if gesture_names:
                    display_frame = overlay_gestures(
                        display_frame,
                        prettify_labels(gesture_names),
                        origin=(10, 110),
                    )
                if status_messages:
                    display_frame = overlay_gestures(
                        display_frame,
                        status_messages,
                        origin=(10, 140 + 30 * len(gesture_names)),
                    )

            if current_mode == Mode.REVIEW:
                display_frame = draw_review_prompts(
                    display_frame,
                    [
                        "Thumb up: Save",
                        "Thumb down: Discard",
                        "Open palm: Back to Gesture Mode",
                        "Three-finger: Exit Gesture Mode",
                    ],
                    origin=(10, 120),
                )

            zoomed = apply_digital_zoom(display_frame, controller.get_zoom_level())
            cv2.imshow("Gesture Camera Controller", zoomed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


