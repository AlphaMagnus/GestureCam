"""Camera action controller for gesture-triggered behaviors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

@dataclass
class CameraState:
    """Mutable state tracked across gesture events."""

    flash_enabled: bool = False
    zoom_level: float = 1.0
    photo_mode: str = "portrait"
    photo_modes: tuple[str, ...] = ("portrait", "landscape", "night")
    zoom_levels: tuple[float, ...] = (1.0, 1.5, 2.0)
    zoom_index: int = 0


class CameraController:
    """Coordinates camera-like operations driven by user gestures."""

    def __init__(self, output_dir: Path | str = "captures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = CameraState()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_timestamp: Optional[float] = None
        self._last_saved_path: Optional[Path] = None

    def get_zoom_level(self) -> float:
        return self.state.zoom_levels[self.state.zoom_index]

    def zoom_in(self) -> str:
        self.state.zoom_index = min(
            self.state.zoom_index + 1, len(self.state.zoom_levels) - 1
        )
        level = self.get_zoom_level()
        return f"Zoom in to x{level:.1f}"

    def zoom_out(self) -> str:
        self.state.zoom_index = max(self.state.zoom_index - 1, 0)
        level = self.get_zoom_level()
        return f"Zoom out to x{level:.1f}"

    def store_capture(self, frame: np.ndarray, timestamp: float) -> None:
        """Hold a frame in memory for review before saving."""

        self._pending_frame = frame.copy()
        self._pending_timestamp = timestamp
        self._last_saved_path = None

    def has_pending_capture(self) -> bool:
        return self._pending_frame is not None

    def peek_pending_capture(self) -> Optional[np.ndarray]:
        if self._pending_frame is None:
            return None
        return self._pending_frame.copy()

    def save_pending_capture(self) -> Optional[Path]:
        if self._pending_frame is None or self._pending_timestamp is None:
            return None

        path = self._save_frame(self._pending_frame, self._pending_timestamp, True)
        self._pending_frame = None
        self._pending_timestamp = None
        self._last_saved_path = path
        return path

    def discard_pending_capture(self) -> None:
        self._pending_frame = None
        self._pending_timestamp = None

    # Internal helpers -------------------------------------------------

    def _save_frame(
        self, frame: np.ndarray, timestamp: float, from_countdown: bool
    ) -> Path:
        timestamp_struct = time.localtime(timestamp)
        fractional_ms = int((timestamp - int(timestamp)) * 1000)
        filename = time.strftime("%Y%m%d_%H%M%S", timestamp_struct)
        filename += f"_{fractional_ms:03d}"
        if from_countdown:
            filename += "_timer"
        path = self.output_dir / f"photo_{filename}.png"
        success = cv2.imwrite(str(path), frame)
        if not success:
            raise RuntimeError("Failed to save photo")
        return path


