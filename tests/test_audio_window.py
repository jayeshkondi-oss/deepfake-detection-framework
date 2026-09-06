import torch
import pytest

from deepfake.data.audio_window import AudioWindowExtractor


def test_creation():
    extractor = AudioWindowExtractor(
        sample_rate=16000,
        duration_seconds=4.0,
    )

    assert extractor.sample_rate == 16000
    assert extractor.num_samples == 64000


def test_invalid_duration():
    with pytest.raises(ValueError):
        AudioWindowExtractor(
            sample_rate=16000,
            duration_seconds=0,
        )


def test_extract_real_video(sample_video):
    extractor = AudioWindowExtractor(
        sample_rate=16000,
        duration_seconds=4.0,
    )

    waveform = extractor.extract(sample_video)

    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.shape[0] == 64000


def test_audio_is_finite(sample_video):
    extractor = AudioWindowExtractor(
        sample_rate=16000,
        duration_seconds=4.0,
    )

    waveform = extractor.extract(sample_video)

    assert torch.isfinite(waveform).all()
