"""
preprocessing.py
Convert raw OpenCV (BGR, uint8) frames into normalized model-ready tensors.
"""

import cv2
import numpy as np
import torch


class FramePreprocessor:
    """
    Resize, convert color space, and normalize frames for the
    visual encoder (ViT expects RGB, float32, ImageNet normalization).
    """

    IMAGENET_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_STD = (0.5, 0.5, 0.5)

    def __init__(self, image_size: int = 224):
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

        self.image_size = image_size

        self._mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single BGR frame.

        Args:
            frame: numpy array [H, W, 3], BGR, uint8.

        Returns:
            Tensor [3, image_size, image_size], float32, normalized.
        """
        if frame is None:
            raise ValueError("frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape [H, W, 3].")

        resized = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self._mean) / self._std

        return tensor

    def process_frames(self, frames: list) -> torch.Tensor:
        """
        Preprocess a list of BGR frames.

        Args:
            frames: list of numpy arrays, each [H, W, 3].

        Returns:
            Tensor [N, 3, image_size, image_size], float32.
        """
        if not frames:
            raise ValueError("frames cannot be empty.")

        return torch.stack(
            [self.process_frame(frame) for frame in frames]
        )
