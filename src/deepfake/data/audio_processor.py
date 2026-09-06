"""
audio_processor.py
Load, convert, resample, and normalize audio for the Wav2Vec2 audio encoder.
"""

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


class AudioProcessor:
    """
    End-to-end audio loading/preprocessing pipeline:

        file -> waveform (as-is) -> mono -> resampled -> normalized
    """

    def __init__(self, sample_rate: int = 16000, normalize: bool = True):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        self.sample_rate = sample_rate
        self.normalize = normalize

    def load_audio(self, path: str | Path) -> tuple[torch.Tensor, int]:
        """
        Load an audio file as-is (no resampling/mono conversion yet).

        Returns:
            waveform: Tensor [channels, samples]
            sample_rate: the file's native sample rate
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)

        # soundfile returns [samples, channels] -> we want [channels, samples]
        waveform = torch.from_numpy(data).transpose(0, 1).contiguous()

        return waveform, sample_rate

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Average all channels down to a single mono channel.

        Args:
            waveform: Tensor [channels, samples]

        Returns:
            Tensor [samples]
        """
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [channels, samples].")

        return waveform.mean(dim=0)

    def resample(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        """
        Resample a mono waveform to self.sample_rate.

        Args:
            waveform: Tensor [samples]
            orig_sample_rate: the waveform's current sample rate

        Returns:
            Tensor [new_samples] at self.sample_rate
        """
        if waveform.ndim != 1:
            raise ValueError("waveform must be 1-D [samples] for resampling.")

        if orig_sample_rate == self.sample_rate:
            return waveform

        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=self.sample_rate,
        )

        return resampler(waveform.unsqueeze(0)).squeeze(0)

    def normalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Peak-normalize a waveform to [-1, 1]."""
        peak = waveform.abs().max()

        if peak > 0:
            waveform = waveform / peak

        return waveform

    def process_file(self, path: str | Path) -> torch.Tensor:
        """
        Full pipeline: load -> mono -> resample -> (optionally) normalize.

        Returns:
            Tensor [samples] at self.sample_rate, float32.
        """
        waveform, sample_rate = self.load_audio(path)
        mono = self.convert_to_mono(waveform)
        resampled = self.resample(mono, sample_rate)

        if self.normalize:
            resampled = self.normalize_waveform(resampled)

        return resampled.float()

    def pad_or_trim(self, waveform: torch.Tensor, length: int) -> torch.Tensor:
        """
        Force a 1-D waveform to an exact number of samples,
        zero-padding if short or truncating if long.
        """
        if waveform.ndim != 1:
            raise ValueError("waveform must be 1-D [samples].")

        current_length = waveform.shape[0]

        if current_length == length:
            return waveform

        if current_length > length:
            return waveform[:length]

        padding = torch.zeros(
            length - current_length,
            dtype=waveform.dtype,
        )
        return torch.cat([waveform, padding])
