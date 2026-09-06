import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal encoder.

    Input:
        [B, T, D]

        B = batch size
        T = number of frames
        D = feature dimension

    Output:
        [B, D]
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_sequence_length: int = 16,
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

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be greater than zero."
            )

        self.feature_dim = feature_dim
        self.max_sequence_length = max_sequence_length

        # Learnable positional embeddings.
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                max_sequence_length,
                feature_dim,
            ) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        return_sequence: bool = False,
    ):
        """
        Encode temporal visual features.

        Args:
            features:
                Tensor with shape [B, T, D].
            return_sequence:
                If True, also return the full per-timestep encoded
                sequence (before pooling), shape [B, T, D] -- useful
                for temporal localization, since the classification
                head can be applied to each timestep independently
                using the same trained weights.

        Returns:
            Tensor with shape [B, D] if return_sequence is False.
            Tuple (pooled [B, D], sequence [B, T, D]) if True.
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "features must be a PyTorch tensor."
            )

        if features.ndim != 3:
            raise ValueError(
                "features must have shape [B, T, D]."
            )

        batch_size, sequence_length, feature_dim = (
            features.shape
        )

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.feature_dim}, "
                f"got {feature_dim}."
            )

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds maximum supported length "
                f"{self.max_sequence_length}."
            )

        # Add positional information.
        positions = self.position_embedding[
            :, :sequence_length, :
        ]

        features = features + positions

        # Temporal Transformer.
        encoded = self.transformer(features)

        # Normalize.
        encoded = self.norm(encoded)

        # Mean pooling across time.
        temporal_features = encoded.mean(dim=1)

        if return_sequence:
            return temporal_features, encoded

        return temporal_features