import pytest
import torch

from deepfake.data.video_audio_extractor import VideoAudioExtractor


def test_extractor_creation():
    extractor = VideoAudioExtractor()

    assert extractor.sample_rate == 16000
    assert extractor.ffmpeg_path == "ffmpeg"


def test_video_validation(sample_video):
    extractor = VideoAudioExtractor()

    validated = extractor._validate_video(sample_video)

    assert validated == sample_video


def test_extract_audio(sample_video, tmp_path):
    extractor = VideoAudioExtractor()

    output_path = tmp_path / "test_extracted_audio.wav"

    result = extractor.extract_audio(
        sample_video,
        output_path,
    )

    assert result.exists()
    assert result == output_path


def test_process_video(sample_video):
    extractor = VideoAudioExtractor()

    waveform = extractor.process_video(sample_video)

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert waveform.numel() > 0


def test_unsupported_video_format(tmp_path):
    extractor = VideoAudioExtractor()

    invalid_path = tmp_path / "test_video.txt"
    invalid_path.write_text("not a video")

    with pytest.raises(ValueError):
        extractor._validate_video(invalid_path)
