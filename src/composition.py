from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from detection import Detection


@dataclass
class CompositionMetadata:
    frame_size: Tuple[int, int]
    thirds_points: Tuple[Tuple[int, int], Tuple[int, int]]
    hint: str
    subject_box: Optional[Tuple[int, int, int, int]]
    subject_center: Optional[Tuple[int, int]]
    target_zoom_box: Optional[Tuple[int, int, int, int]]


def generate_composition_hint(frame: np.ndarray, detection: Optional[Detection]) -> Tuple[str, CompositionMetadata]:
    height, width = frame.shape[:2]
    vertical_third = width // 3
    horizontal_third = height // 3
    thirds_points = ((vertical_third, horizontal_third), (2 * vertical_third, 2 * horizontal_third))

    if detection is None:
        metadata = CompositionMetadata(
            frame_size=(width, height),
            thirds_points=thirds_points,
            hint="No subject detected",
            subject_box=None,
            subject_center=None,
            target_zoom_box=None,
        )
        return metadata.hint, metadata

    x, y, w, h = detection.box
    subject_center = (x + w // 2, y + h // 2)
    frame_center = (width // 2, height // 2)

    hint = "Hold steady"

    dx = subject_center[0] - frame_center[0]
    dy = subject_center[1] - frame_center[1]

    horizontal_hint = ""
    vertical_hint = ""

    threshold_x = width * 0.05
    threshold_y = height * 0.05

    if dx > threshold_x:
        horizontal_hint = "move left"
    elif dx < -threshold_x:
        horizontal_hint = "move right"

    if dy > threshold_y:
        vertical_hint = "move up"
    elif dy < -threshold_y:
        vertical_hint = "move down"

    hint_parts = [part for part in (vertical_hint, horizontal_hint) if part]
    if hint_parts:
        hint = "Please " + " and ".join(hint_parts)
    else:
        # Encourage aligning with rule of thirds
        third_x = vertical_third if subject_center[0] < frame_center[0] else 2 * vertical_third
        if abs(subject_center[0] - third_x) > threshold_x:
            direction = "left" if subject_center[0] > third_x else "right"
            hint = f"Slide {direction} toward rule-of-thirds"

    # Determine zoom guidance based on subject size
    frame_area = width * height
    subject_area = w * h
    area_ratio = subject_area / frame_area

    target_zoom_box: Optional[Tuple[int, int, int, int]] = None

    if area_ratio < 0.05:
        # Too small, zoom in
        zoom_factor = 0.6
        target_zoom_box = _compute_zoom_box(subject_center, frame_size=(width, height), scale=zoom_factor)
        if hint_parts:
            hint += ", plus zooming in"
        else:
            hint = "Zooming in"
    elif area_ratio > 0.25:
        # Too large, zoom out by returning to full frame
        target_zoom_box = (0, 0, width, height)
        hint = "Zooming out for better framing"

    metadata = CompositionMetadata(
        frame_size=(width, height),
        thirds_points=thirds_points,
        hint=hint,
        subject_box=detection.box,
        subject_center=subject_center,
        target_zoom_box=target_zoom_box,
    )

    return hint, metadata


def _compute_zoom_box(subject_center: Tuple[int, int], frame_size: Tuple[int, int], scale: float) -> Tuple[int, int, int, int]:
    width, height = frame_size
    target_width = int(width * scale)
    target_height = int(height * scale)

    cx, cy = subject_center
    x1 = max(0, cx - target_width // 2)
    y1 = max(0, cy - target_height // 2)
    x2 = min(width, x1 + target_width)
    y2 = min(height, y1 + target_height)

    x1 = max(0, x2 - target_width)
    y1 = max(0, y2 - target_height)

    return x1, y1, x2 - x1, y2 - y1

