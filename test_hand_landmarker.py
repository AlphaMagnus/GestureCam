"""Standalone script to validate MediaPipe Hand Landmarker on a still image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.drawing import draw_landmarks


def annotate_image(
    image_path: str,
    model_path: str = "models/hand_landmarker.task",
    output_path: str | None = None,
) -> Path:
    """Run the hand landmarker on a still image and save an annotated copy."""

    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path_obj}")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            "Model file not found. Ensure 'hand_landmarker.task' is present at: "
            f"{model_path_obj}"
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path_obj))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    mp_image = mp.Image.create_from_file(str(image_path_obj))
    detection_result = detector.detect(mp_image)
    detector.close()

    frame_rgb = mp_image.numpy_view()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    landmark_sets = [
        [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        for hand_landmarks in detection_result.hand_landmarks
    ]

    annotated = draw_landmarks(frame_bgr, landmark_sets)

    output_path_obj = (
        Path(output_path)
        if output_path
        else image_path_obj.with_name(f"annotated_{image_path_obj.name}")
    )
    cv2.imwrite(str(output_path_obj), annotated)

    cv2.imshow("Hand Landmarks", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the input image (e.g. image.jpg)")
    parser.add_argument(
        "--model",
        default="models/hand_landmarker.task",
        help="Path to the MediaPipe hand landmarker model file.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the annotated output image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = annotate_image(args.image, args.model, args.output)
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()


