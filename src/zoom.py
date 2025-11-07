from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from composition import CompositionMetadata


class _ZoomState:
    def __init__(self) -> None:
        self.current_box: Optional[Tuple[int, int, int, int]] = None


_state = _ZoomState()


def apply_dynamic_zoom(frame: np.ndarray, metadata: CompositionMetadata) -> np.ndarray:
    target_box = metadata.target_zoom_box
    height, width = frame.shape[:2]

    if target_box is None:
        _state.current_box = (0, 0, width, height)
        return frame

    if _state.current_box is None:
        _state.current_box = target_box
    else:
        _state.current_box = _interpolate_boxes(_state.current_box, target_box, alpha=0.2)

    x, y, w, h = _state.current_box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))

    cropped = frame[y : y + h, x : x + w]
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return zoomed


def _interpolate_boxes(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    nx = (1 - alpha) * ax + alpha * bx
    ny = (1 - alpha) * ay + alpha * by
    nw = (1 - alpha) * aw + alpha * bw
    nh = (1 - alpha) * ah + alpha * bh

    return int(nx), int(ny), int(nw), int(nh)

