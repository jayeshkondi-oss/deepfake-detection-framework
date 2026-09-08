"""
cached_dataset.py
A fast drop-in replacement for DeepfakeDataset that reads pre-extracted
frames and audio from .pt cache files instead of running face detection
on every __getitem__ call.

Use preprocess_cache.py to generate the cache once, then train with
this dataset for dramatically faster epoch times.

Expected cache layout:
    cache_dir/
    +-- real/
    |   +-- video1.pt   -> {"frames": [T,3,H,W], "audio": [S]}
    |   ...
    +-- fake/
        +-- video2.pt
        ...
"""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

LABEL_TO_IDX = {"real": 0, "fake": 1}


class CachedDeepfakeDataset(Dataset):
    """
    Reads pre-computed {frames, audio} tensors from disk.
    Applies optional augmentation (flip, brightness, contrast, rotation)
    at load time -- much faster than re-running face detection each epoch.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        augment: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.augment = augment

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.samples: list[tuple[Path, int]] = []
        for label_name, label_idx in LABEL_TO_IDX.items():
            label_dir = self.cache_dir / label_name
            if not label_dir.exists():
                continue
            for pt_path in sorted(label_dir.glob("*.pt")):
                self.samples.append((pt_path, label_idx))

        if not self.samples:
            raise ValueError(f"No .pt cache files found under {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_label(self, index: int) -> int:
        _, label = self.samples[index]
        return label

    def _augment_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        do_flip = random.random() < 0.5
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        do_rotate = random.random() < 0.5
        angle = random.uniform(-10, 10)

        augmented = []
        for frame in frames_tensor:
            if do_flip:
                frame = TF.hflip(frame)
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            if do_rotate:
                frame = TF.rotate(frame, angle)
            augmented.append(frame)
        return torch.stack(augmented)

    def __getitem__(self, index: int) -> dict:
        pt_path, label = self.samples[index]

        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        frames_tensor = data["frames"]   # [T, 3, H, W]
        audio_tensor = data["audio"]     # [S]

        if self.augment:
            frames_tensor = self._augment_frames(frames_tensor)

        return {
            "frames": frames_tensor,
            "audio": audio_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def close(self):
        pass  # No resources to release
