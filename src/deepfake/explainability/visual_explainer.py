"""
visual_explainer.py
Visual explainability for the ViT-based visual encoder, using
Attention Rollout (Abnar & Zuidema, 2020): attention weights are
combined across all transformer layers to produce a saliency map
showing which image regions the model attended to most. Works
directly off the forward pass (no gradients needed), so it's cheap
and doesn't require touching training code.
"""

import torch

from deepfake.models.visual_encoder import VisualEncoder


class VisualExplainer:
    """
    Produces attention-rollout saliency maps for the ViT visual
    encoder, showing which regions of each face crop the model
    attended to most when forming its representation.
    """

    def __init__(self, visual_encoder: VisualEncoder):
        self.visual_encoder = visual_encoder
        self.patch_size = visual_encoder.vit.config.patch_size
        self.image_size = visual_encoder.vit.config.image_size
        self.grid_size = self.image_size // self.patch_size

    @staticmethod
    def _attention_rollout(attentions: tuple) -> torch.Tensor:
        """
        Combine per-layer attention matrices into a single rollout
        matrix, accounting for the residual connection around each
        attention block.

        Args:
            attentions: tuple of [B, num_heads, N, N] tensors, one
                per transformer layer.

        Returns:
            Tensor [B, N, N].
        """
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
    def explain(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor [T, 3, H, W] or [3, H, W] -- one or more
                already-preprocessed face-crop frames.

        Returns:
            Tensor [T, grid_size, grid_size] (or [grid_size, grid_size]
            for single-frame input), values normalized to [0, 1] per
            frame. Higher values mean the model attended to that
            region more when forming its representation.
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a PyTorch tensor.")

        single_frame = frames.ndim == 3
        if single_frame:
            frames = frames.unsqueeze(0)

        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                "frames must have shape [T, 3, H, W] or [3, H, W]."
            )

        was_training = self.visual_encoder.training
        self.visual_encoder.eval()

        outputs = self.visual_encoder.vit(
            pixel_values=frames,
            output_attentions=True,
        )

        if was_training:
            self.visual_encoder.train()

        rollout = self._attention_rollout(outputs.attentions)

        # CLS token (index 0) attending to each patch token.
        cls_to_patches = rollout[:, 0, 1:]

        # Normalize each frame's map independently to [0, 1].
        min_vals = cls_to_patches.min(dim=1, keepdim=True).values
        max_vals = cls_to_patches.max(dim=1, keepdim=True).values
        normalized = (cls_to_patches - min_vals) / (
            (max_vals - min_vals).clamp(min=1e-8)
        )

        saliency = normalized.reshape(-1, self.grid_size, self.grid_size)

        return saliency.squeeze(0) if single_frame else saliency
