"""Helper functions for drawing hand landmarks and overlays."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm connections
)


def draw_landmarks(
    frame: np.ndarray,
    hand_landmarks: Iterable[Sequence[Tuple[float, float, float]]],
) -> np.ndarray:
    """Render MediaPipe-style landmark points on the frame."""

    output = frame.copy()
    height, width = output.shape[:2]

    for landmarks in hand_landmarks:
        points = [
            (int(x * width), int(y * height))
            for x, y, _ in landmarks
        ]

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(output, points[start], points[end], (0, 255, 0), 2)

        for point in points:
            cv2.circle(output, point, 4, (255, 0, 0), -1)

    return output


def overlay_gestures(
    frame: np.ndarray,
    gestures: Sequence[str],
    origin: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw gesture labels on the frame at a fixed origin."""

    output = frame.copy()
    x, y = origin
    for gesture in gestures:
        cv2.putText(output, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += 30
    return output


def draw_mode_banner(
    frame: np.ndarray,
    text: str,
    *,
    color: Tuple[int, int, int] = (48, 63, 159),
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay a semi-transparent banner at the top-left with mode text."""

    output = frame.copy()
    padding = 12
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    width = text_size[0] + padding * 2
    height = text_size[1] + padding * 2

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.putText(
        output,
        text,
        (padding, padding + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return output


def draw_countdown(frame: np.ndarray, remaining: float) -> np.ndarray:
    """Draw a prominent countdown number at the center of the frame."""

    output = frame.copy()
    height, width = output.shape[:2]
    seconds = max(int(round(remaining)), 0)
    text = str(seconds)
    font_scale = min(width, height) / 250
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
    origin = (
        (width - text_size[0]) // 2,
        (height + text_size[1]) // 2,
    )
    cv2.putText(
        output,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 215, 255),
        4,
    )
    return output


def draw_review_prompts(
    frame: np.ndarray,
    prompts: Sequence[str],
    origin: Tuple[int, int] = (10, 40),
) -> np.ndarray:
    """Display review action prompts on the captured frame."""

    output = frame.copy()
    x, y = origin
    for prompt in prompts:
        cv2.putText(output, prompt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 127), 2)
        y += 35
    return output


def draw_hold_progress(
    frame: np.ndarray,
    progress: float,
    center: Tuple[int, int] | None = None,
    radius: int = 45,
) -> np.ndarray:
    """Draw a circular progress indicator representing gesture hold duration."""

    progress = float(np.clip(progress, 0.0, 1.0))
    output = frame.copy()
    height, width = output.shape[:2]
    if center is None:
        center = (width - radius - 20, radius + 20)

    base_color = (80, 80, 80)
    progress_color = (0, 200, 255)

    cv2.circle(output, center, radius, base_color, 4)

    if progress > 0:
        start_angle = -90
        end_angle = start_angle + int(progress * 360)
        cv2.ellipse(
            output,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            progress_color,
            6,
            lineType=cv2.LINE_AA,
        )

    label = f"{int(progress * 100):d}%"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_origin = (
        center[0] - text_size[0] // 2,
        center[1] + text_size[1] // 2,
    )
    cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


