from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deepfake.data.audio_processor import AudioProcessor


@pytest.fixture
def audio_file(tmp_path):
    """
    Create a temporary stereo WAV file.
    """

    sample_rate = 8000
    duration = 1.0

    samples = int(
        sample_rate * duration
    )

    time = np.linspace(
        0,
        duration,
        samples,
        endpoint=False,
    )

    left = 0.5 * np.sin(
        2 * np.pi * 440 * time
    )

    right = 0.25 * np.sin(
        2 * np.pi * 220 * time
    )

    audio = np.stack(
        [left, right],
        axis=1,
    ).astype(np.float32)

    path = tmp_path / "test_audio.wav"

    sf.write(
        str(path),
        audio,
        sample_rate,
    )

    return path


def test_audio_processor_creation():
    processor = AudioProcessor(
        sample_rate=16000
    )

    assert processor.sample_rate == 16000
    assert processor.normalize is True


def test_audio_loading(audio_file):
    processor = AudioProcessor()

    waveform, sample_rate = (
        processor.load_audio(audio_file)
    )

    assert isinstance(
        waveform,
        torch.Tensor,
    )

    assert waveform.ndim == 2
    assert waveform.shape[0] == 2
    assert sample_rate == 8000


def test_mono_conversion(audio_file):
    processor = AudioProcessor()

    waveform, _ = processor.load_audio(
        audio_file
    )

    mono = processor.convert_to_mono(
        waveform
    )

    assert isinstance(
        mono,
        torch.Tensor,
    )

    assert mono.ndim == 1
    assert mono.shape[0] == waveform.shape[1]


def test_resampling(audio_file):
    processor = AudioProcessor(
        sample_rate=16000
    )

    waveform, original_rate = (
        processor.load_audio(audio_file)
    )

    mono = processor.convert_to_mono(
        waveform
    )

    resampled = processor.resample(
        mono,
        original_rate,
    )

    assert resampled.ndim == 1

    # Original: 1 second at 8 kHz.
    # Target: 1 second at 16 kHz.
    assert resampled.shape[0] == 16000


def test_process_file(audio_file):
    processor = AudioProcessor(
        sample_rate=16000
    )

    waveform = processor.process_file(
        audio_file
    )

    assert isinstance(
        waveform,
        torch.Tensor,
    )

    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.shape[0] == 16000

    assert waveform.abs().max() <= 1.0 + 1e-6


def test_pad_or_trim():
    processor = AudioProcessor()

    waveform = torch.ones(1000)

    padded = processor.pad_or_trim(
        waveform,
        16000,
    )

    assert padded.shape == (16000,)
    assert torch.allclose(
        padded[:1000],
        waveform,
    )

    trimmed = processor.pad_or_trim(
        waveform,
        500,
    )

    assert trimmed.shape == (500,)
    assert torch.allclose(
        trimmed,
        waveform[:500],
    )