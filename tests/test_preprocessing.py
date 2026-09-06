import cv2
import numpy as np
import pytest
import torch

from deepfake.data.preprocessing import FramePreprocessor


def create_test_frame():
    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )


def test_single_frame_preprocessing():
    preprocessor = FramePreprocessor(
        image_size=224
    )

    frame = create_test_frame()

    tensor = preprocessor.process_frame(frame)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_multiple_frame_preprocessing():
    preprocessor = FramePreprocessor(
        image_size=224
    )

    frames = [
        create_test_frame()
        for _ in range(16)
    ]

    tensor = preprocessor.process_frames(frames)

    assert tensor.shape == (
        16,
        3,
        224,
        224,
    )

    assert tensor.dtype == torch.float32


def test_custom_image_size():
    preprocessor = FramePreprocessor(
        image_size=128
    )

    frame = create_test_frame()

    tensor = preprocessor.process_frame(frame)

    assert tensor.shape == (3, 128, 128)


def test_empty_frames():
    preprocessor = FramePreprocessor()

    with pytest.raises(ValueError):
        preprocessor.process_frames([])


def test_invalid_frame():
    preprocessor = FramePreprocessor()

    with pytest.raises(ValueError):
        preprocessor.process_frame(None)