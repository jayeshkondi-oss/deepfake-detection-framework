"""
temporal_explainer.py
Per-frame "when is it fake" localization. Reuses the model's already-
trained fusion and classification head -- applied to each frame's
temporal representation individually instead of only to the pooled
whole-video representation -- so this requires zero additional
training. Only the small fusion + classifier modules run once per
frame; the heavy visual/audio encoders each still run exactly once.
"""

import torch

from deepfake.models.deepfake_detector import DeepfakeDetector


class TemporalExplainer:
    """
    Produces a per-frame FAKE-probability timeline, showing which
    parts of the video the model finds most suspicious.
    """

    def __init__(self, model: DeepfakeDetector):
        self.model = model

    @torch.no_grad()
    def explain(
        self,
        frames: torch.Tensor,
        waveform: torch.Tensor,
        fps: float | None = None,
    ) -> dict:
        """
        Args:
            frames: Tensor [T, 3, H, W] -- one video's preprocessed frames.
            waveform: Tensor [samples] -- that video's audio.
            fps: optional effective frame rate (sampled frames per
                second of original video), used to convert frame
                indices into timestamps in the returned dict.

        Returns:
            {
                "fake_probability_per_frame": Tensor [T],
                "times_seconds": Tensor [T], only present if fps
                    was given.
            }
        """
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise ValueError("frames must have shape [T, 3, H, W].")

        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1:
            raise ValueError("waveform must have shape [samples].")

        was_training = self.model.training
        self.model.eval()

        num_frames = frames.shape[0]

        # Visual features for every frame, computed once.
        visual_features = self.model.visual_encoder(frames)
        visual_features = visual_features.unsqueeze(0)  # [1, T, D]

        # Per-timestep temporal encoding (before pooling), once.
        _, sequence = self.model.temporal_transformer(
            visual_features, return_sequence=True
        )  # sequence: [1, T, D]

        # Audio features, computed once -- shared across all frames,
        # since this architecture doesn't segment audio per-frame.
        audio_features = self.model.audio_encoder(waveform.unsqueeze(0))
        audio_features = audio_features.unsqueeze(1)  # [1, 1, D]

        fake_probabilities = []
        for t in range(num_frames):
            frame_repr = sequence[:, t : t + 1, :]  # [1, 1, D]
            fused = self.model.fusion(frame_repr, audio_features)  # [1, D]
            logits = self.model.classifier(fused)  # [1, 2]
            probs = torch.softmax(logits, dim=-1)
            fake_probabilities.append(probs[0, 1].item())

        if was_training:
            self.model.train()

        result = {
            "fake_probability_per_frame": torch.tensor(fake_probabilities),
        }

        if fps is not None and fps > 0:
            result["times_seconds"] = (
                torch.arange(num_frames, dtype=torch.float32) / fps
            )

        return result
