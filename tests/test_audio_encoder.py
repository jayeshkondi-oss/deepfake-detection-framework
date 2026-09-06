import pytest
import torch

from deepfake.models.audio_encoder import AudioEncoder


def test_audio_encoder_creation():
    encoder = AudioEncoder(
        freeze_backbone=True
    )

    assert encoder is not None
    assert encoder.feature_dim == 768


def test_audio_encoder_output_shape():
    encoder = AudioEncoder(
        freeze_backbone=True
    )

    # Approximately 1 second at 16 kHz.
    waveform = torch.randn(
        2,
        16000,
    )

    with torch.no_grad():
        features = encoder(waveform)

    assert features.shape == (2, 768)


def test_audio_encoder_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = torch.device("cuda:0")

    encoder = AudioEncoder(
        freeze_backbone=True
    ).to(device)

    waveform = torch.randn(
        1,
        16000,
        device=device,
    )

    with torch.no_grad():
        features = encoder(waveform)

    assert features.shape == (1, 768)
    assert features.device.type == "cuda"


def test_invalid_audio_dimensions():
    encoder = AudioEncoder(
        freeze_backbone=True
    )

    waveform = torch.randn(
        16000,
    )

    with pytest.raises(ValueError):
        encoder(waveform)