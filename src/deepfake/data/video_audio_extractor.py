"""
video_audio_extractor.py
Extract the audio track from a video file using ffmpeg, and load it
straight into a waveform tensor.
"""

import subprocess
import tempfile
from pathlib import Path

import torch

from deepfake.data.audio_processor import AudioProcessor

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoAudioExtractor:
    """
    Pulls the audio track out of a video file (via ffmpeg) and hands
    back a mono, resampled waveform tensor ready for the audio encoder.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        ffmpeg_path: str = "ffmpeg",
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        self.sample_rate = sample_rate
        self.ffmpeg_path = ffmpeg_path
        self._audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            normalize=True,
        )

    def _validate_video(self, video_path: str | Path) -> Path:
        """
        Check that a path looks like a supported video file by
        extension. Does NOT check that the file exists -- callers
        that need the file to exist (extract_audio, process_video)
        check that separately, since validation-by-extension is
        useful even for paths that don't exist yet.
        """
        video_path = Path(video_path)

        if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{video_path.suffix}'. "
                f"Supported formats: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

        return video_path

    def extract_audio(
        self,
        video_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Extract the audio track from video_path into a .wav file at
        output_path, resampled to self.sample_rate, mono.

        Returns:
            output_path, as a Path.
        """
        video_path = self._validate_video(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self.ffmpeg_path,
            "-y",  # overwrite output without prompting
            "-i", str(video_path),
            "-vn",  # no video
            "-ac", "1",  # mono
            "-ar", str(self.sample_rate),
            "-f", "wav",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract audio from {video_path}:\n"
                f"{result.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"ffmpeg reported success but no output was written to "
                f"{output_path}"
            )

        return output_path

    def process_video(self, video_path: str | Path) -> torch.Tensor:
        """
        Extract audio from a video and return it directly as a
        normalized, mono, resampled waveform tensor -- no output
        file left behind.

        Returns:
            Tensor [samples], float32.
        """
        video_path = self._validate_video(video_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_wav = Path(tmp_dir) / "extracted_audio.wav"
            self.extract_audio(video_path, tmp_wav)
            return self._audio_processor.process_file(tmp_wav)
