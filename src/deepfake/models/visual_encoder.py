import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


class VisualEncoder(nn.Module):
    """
    Vision Transformer (ViT) visual encoder.

    Input:
        [B, 3, 224, 224]

    Output:
        [B, 768]
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        if pretrained:
            self.vit = ViTModel.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        else:
            # Random-init ViT with the vit-base-patch16-224
            # architecture, with no network access required.
            # Used for fast/offline tests and architecture checks
            # where pretrained weights don't matter.
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
                attn_implementation="eager",
            )
            self.vit = ViTModel(config)

        self.feature_dim = self.vit.config.hidden_size

        if freeze_backbone:
            # Freeze all ViT parameters first.
            for parameter in self.vit.parameters():
                parameter.requires_grad = False

        # Optionally unfreeze the last N encoder layers so they can
        # adapt to deepfake-specific features while the early layers
        # (general edge/texture detectors) stay frozen.
        self._unfreeze_last_n = 0

    def unfreeze_last_n_layers(self, n: int) -> None:
        """Unfreeze the last N transformer encoder blocks + final LayerNorm."""
        if n <= 0:
            return
        self._unfreeze_last_n = n
        encoder_layers = self.vit.encoder.layer
        total = len(encoder_layers)
        unfreeze_from = max(0, total - n)
        unfrozen_params = 0
        for i, layer in enumerate(encoder_layers):
            if i >= unfreeze_from:
                for p in layer.parameters():
                    p.requires_grad = True
                    unfrozen_params += p.numel()
        # Also unfreeze the final LayerNorm.
        for p in self.vit.layernorm.parameters():
            p.requires_grad = True
            unfrozen_params += p.numel()
        print(
            f"VisualEncoder: unfroze last {n}/{total} ViT blocks "
            f"({unfrozen_params:,} params)"
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract visual features from images.

        Args:
            images:
                Tensor of shape [B, 3, 224, 224].

        Returns:
            Tensor of shape [B, 768].
        """

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "images must be a PyTorch tensor."
            )

        if images.ndim != 4:
            raise ValueError(
                "images must have shape [B, 3, H, W]."
            )

        if images.shape[1] != 3:
            raise ValueError(
                "images must have 3 color channels."
            )

        outputs = self.vit(
            pixel_values=images
        )

        # CLS token representation.
        features = outputs.last_hidden_state[:, 0]

        return features