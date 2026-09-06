import torch
import torch.nn as nn

from deepfake.data.preprocessing import FramePreprocessor
from deepfake.models.visual_encoder import VisualEncoder


class VisualFeatureExtractor(nn.Module):
    """
    Extract visual features from a sequence of video frames.

    Pipeline:

        Frames
          ↓
        Preprocessing
          ↓
        Vision Transformer
          ↓
        768-dimensional features

    Input:
        [T, 3, 224, 224]

    Output:
        [T, 768]
    """

    def __init__(
        self,
        image_size: int = 224,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.preprocessor = FramePreprocessor(
            image_size=image_size
        )

        self.encoder = VisualEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        self.feature_dim = self.encoder.feature_dim

    def forward(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from preprocessed frames.

        Args:
            frames:
                Tensor with shape [T, 3, 224, 224]

        Returns:
            Tensor with shape [T, 768]
        """

        if not isinstance(frames, torch.Tensor):
            raise TypeError(
                "frames must be a PyTorch tensor."
            )

        if frames.ndim != 4:
            raise ValueError(
                "frames must have shape [T, 3, 224, 224]."
            )

        if frames.shape[1] != 3:
            raise ValueError(
                "frames must have 3 channels."
            )

        if (
            frames.shape[2] != 224
            or frames.shape[3] != 224
        ):
            raise ValueError(
                "frames must have spatial size 224x224."
            )

        return self.encoder(frames)

    def extract(
        self,
        frames: list,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Preprocess raw OpenCV frames and extract
        ViT features.

        Args:
            frames:
                List of BGR OpenCV frames.

            device:
                Optional torch device.

        Returns:
            Tensor with shape [T, 768].
        """

        if not frames:
            raise ValueError(
                "frames cannot be empty."
            )

        processed = self.preprocessor.process_frames(
            frames
        )

        if device is not None:
            processed = processed.to(device)

        return self.forward(processed)