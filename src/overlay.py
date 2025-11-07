from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from composition import CompositionMetadata


def draw_overlay(frame: np.ndarray, hint: str, metadata: CompositionMetadata) -> np.ndarray:
    output = frame.copy()
    width, height = metadata.frame_size

    # Draw rule-of-thirds grid
    for x in (metadata.thirds_points[0][0], metadata.thirds_points[1][0]):
        cv2.line(output, (x, 0), (x, height), (0, 255, 0), 1, cv2.LINE_AA)
    for y in (metadata.thirds_points[0][1], metadata.thirds_points[1][1]):
        cv2.line(output, (0, y), (width, y), (0, 255, 0), 1, cv2.LINE_AA)

    if metadata.subject_box is not None:
        x, y, w, h = metadata.subject_box
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 140, 0), 2)
        cv2.circle(output, metadata.subject_center, 5, (255, 140, 0), -1)

    cv2.putText(
        output,
        hint,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    zoom_level = _estimate_zoom_level(metadata, width, height)
    cv2.putText(
        output,
        f"Zoom: {zoom_level:0.0f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    return output


def _estimate_zoom_level(metadata: CompositionMetadata, frame_width: int, frame_height: int) -> float:
    if metadata.target_zoom_box is None:
        return 100.0

    _, _, w, h = metadata.target_zoom_box
    frame_area = frame_width * frame_height
    box_area = w * h
    zoom = frame_area / max(1, box_area) * 100
    return min(400.0, max(50.0, zoom))

