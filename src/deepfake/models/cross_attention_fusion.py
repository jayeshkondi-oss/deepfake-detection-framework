import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of visual and audio features.

    Visual features act as queries.
    Audio features provide keys and values.

    Inputs:
        visual_features:
            [B, T_v, D]

        audio_features:
            [B, T_a, D]

    Output:
        [B, D]
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(
                "feature_dim must be greater than zero."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

        if feature_dim % num_heads != 0:
            raise ValueError(
                "feature_dim must be divisible by num_heads."
            )

        self.feature_dim = feature_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(feature_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse visual and audio representations.

        Args:
            visual_features:
                [B, T_v, D]

            audio_features:
                [B, T_a, D]

        Returns:
            [B, D]
        """

        if not isinstance(
            visual_features,
            torch.Tensor,
        ):
            raise TypeError(
                "visual_features must be a PyTorch tensor."
            )

        if not isinstance(
            audio_features,
            torch.Tensor,
        ):
            raise TypeError(
                "audio_features must be a PyTorch tensor."
            )

        if visual_features.ndim != 3:
            raise ValueError(
                "visual_features must have shape [B, T, D]."
            )

        if audio_features.ndim != 3:
            raise ValueError(
                "audio_features must have shape [B, T, D]."
            )

        if visual_features.shape[0] != audio_features.shape[0]:
            raise ValueError(
                "Visual and audio batch sizes must match."
            )

        if visual_features.shape[2] != self.feature_dim:
            raise ValueError(
                f"Visual feature dimension must be "
                f"{self.feature_dim}."
            )

        if audio_features.shape[2] != self.feature_dim:
            raise ValueError(
                f"Audio feature dimension must be "
                f"{self.feature_dim}."
            )

        # Visual features query the audio representation.
        attended_audio, _ = self.cross_attention(
            query=visual_features,
            key=audio_features,
            value=audio_features,
        )

        # Residual connection + normalization.
        fused = self.norm1(
            visual_features + attended_audio
        )

        # Feed-forward refinement.
        refined = self.feed_forward(fused)

        fused = self.norm2(
            fused + refined
        )

        # Aggregate across visual time steps.
        fused = fused.mean(dim=1)

        return fused