import cv2
import numpy as np
import pytest
import torch

from deepfake.models.visual_feature_extractor import (
    VisualFeatureExtractor,
)


def test_feature_extractor_creation():
    extractor = VisualFeatureExtractor(
        pretrained=False
    )

    assert extractor is not None
    assert extractor.feature_dim == 768


def test_feature_extraction_shape():
    extractor = VisualFeatureExtractor(
        pretrained=False
    )

    frames = torch.randn(
        4,
        3,
        224,
        224,
    )

    with torch.no_grad():
        features = extractor(frames)

    assert features.shape == (4, 768)


def test_feature_extraction_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = torch.device("cuda:0")

    extractor = VisualFeatureExtractor(
        pretrained=False
    ).to(device)

    frames = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        features = extractor(frames)

    assert features.shape == (4, 768)
    assert features.device.type == "cuda"


def test_raw_frame_extraction():
    extractor = VisualFeatureExtractor(
        pretrained=False
    )

    frames = []

    for _ in range(4):
        frame = np.random.randint(
            0,
            256,
            (480, 640, 3),
            dtype=np.uint8,
        )

        frames.append(frame)

    with torch.no_grad():
        features = extractor.extract(frames)

    assert features.shape == (4, 768)


def test_empty_frames():
    extractor = VisualFeatureExtractor(
        pretrained=False
    )

    with pytest.raises(ValueError):
        extractor.extract([])


def test_invalid_tensor_shape():
    extractor = VisualFeatureExtractor(
        pretrained=False
    )

    frames = torch.randn(
        4,
        3,
        128,
        128,
    )

    with pytest.raises(ValueError):
        extractor(frames)