"""
face_detector.py
Face detection using MediaPipe's Face Detector task
(BlazeFace short-range model, models/mediapipe/blaze_face_short_range.tflite).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector as _MPFaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "mediapipe"
    / "blaze_face_short_range.tflite"
)


@dataclass
class FaceDetection:
    """
    A single detected face.

    bbox: (x1, y1, x2, y2) in pixel coordinates.
    confidence: detection confidence in [0, 1].
    """

    bbox: tuple[int, int, int, int]
    confidence: float


class FaceDetector:
    """
    Thin wrapper around MediaPipe's Face Detector task.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        min_confidence: float = 0.5,
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face detector model not found: {model_path}. "
                "Make sure models/mediapipe/blaze_face_short_range.tflite "
                "is present in the project."
            )

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )

        self._detector = _MPFaceDetector.create_from_options(options)
        self._closed = False

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in a frame.

        Args:
            frame: numpy array of shape [H, W, 3], RGB order.

        Returns:
            List of FaceDetection (possibly empty).
        """
        if frame is None:
            raise ValueError("frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape [H, W, 3].")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame),
        )

        result = self._detector.detect(mp_image)

        detections = []
        for detection in result.detections:
            box = detection.bounding_box
            x1 = box.origin_x
            y1 = box.origin_y
            x2 = box.origin_x + box.width
            y2 = box.origin_y + box.height

            confidence = (
                detection.categories[0].score
                if detection.categories
                else 0.0
            )

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )

        return detections

    def crop_face(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        margin: float = 0.2,
    ) -> np.ndarray:
        """
        Crop a detected face out of a frame, with an extra margin
        around the bounding box.

        Args:
            frame: numpy array [H, W, 3].
            detection: a FaceDetection with a bbox.
            margin: fraction of the box size to pad on each side.

        Returns:
            Cropped face as a numpy array [h, w, 3].
        """
        if frame is None:
            raise ValueError("frame cannot be None.")

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox

        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = int(box_w * margin)
        pad_y = int(box_h * margin)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        return frame[y1:y2, x1:x2].copy()

    def close(self):
        """Release the underlying MediaPipe detector."""
        if not self._closed:
            self._detector.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
