from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    confidence: float


def detect_primary_subject(frame: np.ndarray) -> Optional[Detection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    # Choose the largest face as the primary subject
    best_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = best_face
    return Detection(box=(x, y, w, h), confidence=1.0)

