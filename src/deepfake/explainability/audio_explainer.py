"""
audio_explainer.py
Audio explainability for the Wav2Vec2-based audio encoder, using the
same Attention Rollout technique as the visual explainer, adapted for
audio: since Wav2Vec2 has no CLS/summary token, per-timestep
importance is estimated as the average attention each time-frame
receives from all other time-frames after rolling out across layers.
"""

import torch

from deepfake.models.audio_encoder import AudioEncoder


class AudioExplainer:
    """
    Produces an importance-over-time curve for the Wav2Vec2 audio
    encoder, showing which portions of the audio clip the model
    attended to most.
    """

    def __init__(self, audio_encoder: AudioEncoder):
        self.audio_encoder = audio_encoder

    @staticmethod
    def _attention_rollout(attentions: tuple) -> torch.Tensor:
        """Same rollout as VisualExplainer -- see that class for details."""
        batch_size, _, seq_len, _ = attentions[0].shape
        device = attentions[0].device

        identity = torch.eye(seq_len, device=device).unsqueeze(0)
        rollout = identity.expand(batch_size, -1, -1).clone()

        for attention in attentions:
            attention_avg = attention.mean(dim=1)
            attention_avg = attention_avg + identity
            attention_avg = attention_avg / attention_avg.sum(
                dim=-1, keepdim=True
            )
            rollout = torch.bmm(attention_avg, rollout)

        return rollout

    @torch.no_grad()
    def explain(
        self,
        waveform: torch.Tensor,
        duration_seconds: float | None = None,
    ) -> dict:
        """
        Args:
            waveform: Tensor [samples] or [B, samples].
            duration_seconds: optional clip duration, used to convert
                frame indices into timestamps in the returned dict.

        Returns:
            {
                "importance": Tensor [T] (or [B, T] for batched
                    input), normalized to [0, 1],
                "times_seconds": Tensor [T], only present if
                    duration_seconds was given.
            }
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform must be a PyTorch tensor.")

        single = waveform.ndim == 1
        if single:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(
                "waveform must have shape [samples] or [B, samples]."
            )

        was_training = self.audio_encoder.training
        self.audio_encoder.eval()

        outputs = self.audio_encoder.model(
            input_values=waveform,
            output_attentions=True,
        )

        if was_training:
            self.audio_encoder.train()

        rollout = self._attention_rollout(outputs.attentions)

        # Average incoming attention across all query positions --
        # how much each time-frame was attended to overall, after
        # composing attention across every layer.
        importance = rollout.mean(dim=1)

        min_vals = importance.min(dim=1, keepdim=True).values
        max_vals = importance.max(dim=1, keepdim=True).values
        normalized = (importance - min_vals) / (
            (max_vals - min_vals).clamp(min=1e-8)
        )

        result = {
            "importance": normalized.squeeze(0) if single else normalized,
        }

        if duration_seconds is not None:
            num_frames = normalized.shape[1]
            result["times_seconds"] = torch.linspace(
                0, duration_seconds, num_frames
            )

        return result
