import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for binary deepfake detection.

    Input:
        [batch_size, feature_dim]

    Output:
        [batch_size, 2]

    Classes:
        0 = REAL
        1 = FAKE
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if num_classes != 2:
            raise ValueError(
                "This classification head supports exactly 2 classes."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify fused multimodal features.

        Args:
            features:
                Tensor of shape [B, input_dim].

        Returns:
            Logits of shape [B, 2].
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "features must be a PyTorch tensor."
            )

        if features.ndim != 2:
            raise ValueError(
                "features must have shape [batch_size, input_dim]."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, "
                f"got {features.shape[1]}."
            )

        return self.classifier(features)