"""
audio_window.py
Extract a fixed-length audio window (a set number of seconds, as a
set number of samples) directly from a video file, for feeding into
the audio encoder alongside a fixed-length sequence of video frames.
"""

from pathlib import Path

import torch

from deepfake.data.audio_processor import AudioProcessor
from deepfake.data.video_audio_extractor import VideoAudioExtractor


class AudioWindowExtractor:
    """
    Extracts audio from a video and forces it to an exact, fixed
    number of samples (padding with zeros if the clip is shorter,
    trimming if it's longer), so every sample in a batch lines up.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero.")

        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.num_samples = int(sample_rate * duration_seconds)

        self._extractor = VideoAudioExtractor(sample_rate=sample_rate)
        self._audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            normalize=True,
        )

    def extract(self, video_path: str | Path) -> torch.Tensor:
        """
        Args:
            video_path: path to a video file.

        Returns:
            Tensor [num_samples], float32, at self.sample_rate.
        """
        waveform = self._extractor.process_video(video_path)
        return self._audio_processor.pad_or_trim(waveform, self.num_samples)
