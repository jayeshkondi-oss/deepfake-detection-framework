import cv2
import numpy as np
import pytest

from deepfake.data.face_detector import (
    FaceDetection,
    FaceDetector,
)


def create_test_frame():
    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )


def test_detector_creation():
    detector = FaceDetector()

    assert detector is not None

    detector.close()


def test_no_face_detection():
    detector = FaceDetector()

    frame = create_test_frame()

    detections = detector.detect(frame)

    assert isinstance(detections, list)
    assert len(detections) == 0

    detector.close()


def test_invalid_frame():
    detector = FaceDetector()

    with pytest.raises(ValueError):
        detector.detect(None)

    detector.close()


def test_face_detection_dataclass():
    detection = FaceDetection(
        bbox=(10, 20, 100, 120),
        confidence=0.95,
    )

    assert detection.bbox == (
        10,
        20,
        100,
        120,
    )

    assert detection.confidence == 0.95


def test_crop_face():
    detector = FaceDetector()

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detection = FaceDetection(
        bbox=(200, 150, 300, 250),
        confidence=0.95,
    )

    face = detector.crop_face(
        frame,
        detection,
        margin=0.2,
    )

    assert face is not None
    assert face.size > 0
    assert face.shape[2] == 3

    detector.close()