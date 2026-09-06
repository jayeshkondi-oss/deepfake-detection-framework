import pytest
import torch

from deepfake.data.audio_preprocessing import (
    AudioPreprocessor,
)


def test_preprocessor_creation():
    processor = AudioPreprocessor()

    assert processor.sample_rate == 16000


def test_mono_waveform():
    processor = AudioPreprocessor()

    waveform = torch.randn(
        2,
        16000,
    )

    processed = processor.process_waveform(
        waveform,
        16000,
    )

    assert processed.ndim == 1
    assert processed.shape[0] == 16000


def test_resampling():
    processor = AudioPreprocessor()

    # One second of 48 kHz stereo audio.
    waveform = torch.randn(
        2,
        48000,
    )

    processed = processor.process_waveform(
        waveform,
        48000,
    )

    assert processed.ndim == 1
    assert processed.shape[0] == 16000


def test_mono_input():
    processor = AudioPreprocessor()

    waveform = torch.randn(
        16000,
    )

    processed = processor.process_waveform(
        waveform,
        16000,
    )

    assert processed.ndim == 1
    assert processed.shape[0] == 16000


def test_normalization():
    processor = AudioPreprocessor()

    waveform = torch.randn(
        16000,
    ) * 10

    processed = processor.process_waveform(
        waveform,
        16000,
    )

    assert processed.abs().max() <= 1.0


def test_invalid_waveform():
    processor = AudioPreprocessor()

    waveform = torch.randn(
        2,
        3,
        16000,
    )

    with pytest.raises(ValueError):
        processor.process_waveform(
            waveform,
            16000,
        )


def test_invalid_sample_rate():
    with pytest.raises(ValueError):
        AudioPreprocessor(
            sample_rate=0
        )