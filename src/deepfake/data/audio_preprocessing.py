"""
audio_preprocessing.py
Lightweight, in-memory waveform preprocessing: mono conversion,
resampling, and normalization for a waveform you already have loaded
(e.g. audio extracted from a video frame-by-frame elsewhere).

For loading audio directly from a file on disk, see AudioProcessor
in audio_processor.py -- this class delegates its per-waveform logic
to AudioProcessor so the two never drift out of sync.
"""

import torch

from deepfake.data.audio_processor import AudioProcessor


class AudioPreprocessor:
    """
    Convenience wrapper for callers that already have a waveform in
    memory and just need it converted to mono, resampled, and
    normalized in a single call.
    """

    def __init__(self, sample_rate: int = 16000):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        self.sample_rate = sample_rate
        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            normalize=True,
        )

    def process_waveform(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: int,
    ) -> torch.Tensor:
        """
        Args:
            waveform: Tensor [samples] (mono) or [channels, samples].
            orig_sample_rate: the waveform's current sample rate.

        Returns:
            Tensor [new_samples] at self.sample_rate, normalized to [-1, 1].
        """
        if waveform.ndim == 1:
            mono = waveform
        elif waveform.ndim == 2:
            mono = self._processor.convert_to_mono(waveform)
        else:
            raise ValueError(
                "waveform must have shape [samples] or [channels, samples]."
            )

        resampled = self._processor.resample(mono, orig_sample_rate)
        return self._processor.normalize_waveform(resampled)
