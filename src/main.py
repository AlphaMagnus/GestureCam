import cv2

from detection import detect_primary_subject
from composition import generate_composition_hint
from overlay import draw_overlay
from zoom import apply_dynamic_zoom


def webcam_loop(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = detect_primary_subject(frame)
            hint, metadata = generate_composition_hint(frame, detection)
            frame = apply_dynamic_zoom(frame, metadata)
            output = draw_overlay(frame, hint, metadata)

            cv2.imshow("Webcam Composition Assistant", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_loop()

