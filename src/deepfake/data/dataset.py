"""
dataset.py
PyTorch Dataset that walks a real/fake video directory structure and
serves preprocessed face-crop frame tensors AND an audio window ready
for the visual and audio encoders respectively.

Expected directory layout:

    root/
    ├── real/
    │   └── *.mp4
    └── fake/
        └── *.mp4
"""

from pathlib import Path

import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from deepfake.data.audio_window import AudioWindowExtractor
from deepfake.data.face_detector import FaceDetector
from deepfake.data.preprocessing import FramePreprocessor
from deepfake.data.video_loader import VideoLoader

LABEL_TO_IDX = {"real": 0, "fake": 1}


class DeepfakeDataset(Dataset):
    """
    Video-level dataset: each item is one video, represented as a
    fixed-length sequence of preprocessed face-crop frames plus a
    fixed-length audio window -- everything DeepfakeDetector needs.

    Falls back to a preprocessed (but uncropped) full frame whenever
    no face is detected, and falls back to a silent waveform whenever
    a video has no audio track or audio extraction fails, so a
    handful of hard/edge-case samples never crash training -- they
    just carry a weaker signal.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 16,
        image_size: int = 224,
        face_margin: float = 0.2,
        audio_sample_rate: int = 16000,
        audio_duration_seconds: float = 4.0,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        if num_frames <= 0:
            raise ValueError("num_frames must be greater than zero.")

        self.num_frames = num_frames
        self.image_size = image_size
        self.face_margin = face_margin
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration_seconds = audio_duration_seconds
        self.augment = augment

        self.samples: list[tuple[Path, int]] = []
        for label_name, label_idx in LABEL_TO_IDX.items():
            label_dir = self.root_dir / label_name
            if not label_dir.exists():
                continue
            for video_path in sorted(label_dir.glob("*.mp4")):
                self.samples.append((video_path, label_idx))

        self._preprocessor = FramePreprocessor(image_size=image_size)
        self._face_detector: FaceDetector | None = None
        self._audio_extractor: AudioWindowExtractor | None = None

    def _augment_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply consistent random augmentation across all T frames of a video.

        Augmentations applied the same way to every frame so temporal
        structure is preserved:
          - Random horizontal flip (50% chance)
          - Random brightness + contrast jitter (±20%)
          - Random rotation ±10 degrees (50% chance)

        Args:
            frames_tensor: [T, 3, H, W]

        Returns:
            Augmented tensor of the same shape.
        """
        # Decide all random choices once, then apply consistently.
        do_flip = random.random() < 0.5
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        do_rotate = random.random() < 0.5
        angle = random.uniform(-10, 10)

        augmented = []
        for frame in frames_tensor:  # frame: [3, H, W]
            if do_flip:
                frame = TF.hflip(frame)
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            if do_rotate:
                frame = TF.rotate(frame, angle)
            augmented.append(frame)

        return torch.stack(augmented)

    def _get_face_detector(self) -> FaceDetector:
        """Lazily create the face detector (and its MediaPipe model)
        on first use, so simply constructing a Dataset stays cheap."""
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector

    def _get_audio_extractor(self) -> AudioWindowExtractor:
        """Lazily create the audio window extractor on first use."""
        if self._audio_extractor is None:
            self._audio_extractor = AudioWindowExtractor(
                sample_rate=self.audio_sample_rate,
                duration_seconds=self.audio_duration_seconds,
            )
        return self._audio_extractor

    def __len__(self) -> int:
        return len(self.samples)

    def get_label(self, index: int) -> int:
        _, label = self.samples[index]
        return label

    def get_video_path(self, index: int) -> Path:
        path, _ = self.samples[index]
        return path

    def __getitem__(self, index: int) -> dict:
        video_path, label = self.samples[index]

        with VideoLoader(video_path) as loader:
            raw_frames = loader.sample_frames(self.num_frames)

        detector = self._get_face_detector()

        face_frames = []
        for frame in raw_frames:
            # MediaPipe expects RGB; OpenCV frames are BGR.
            rgb_frame = frame[:, :, ::-1]
            detections = detector.detect(rgb_frame)

            if detections:
                best = max(detections, key=lambda d: d.confidence)
                face_frames.append(
                    detector.crop_face(frame, best, margin=self.face_margin)
                )
            else:
                # No face found in this frame -- fall back to the
                # full frame rather than dropping it, so every video
                # still yields exactly num_frames.
                face_frames.append(frame)

        frames_tensor = self._preprocessor.process_frames(face_frames)

        # Apply augmentation only during training (augment=True).
        if self.augment:
            frames_tensor = self._augment_frames(frames_tensor)

        label_tensor = torch.tensor(label, dtype=torch.long)

        try:
            audio_tensor = self._get_audio_extractor().extract(video_path)
        except Exception:
            # No audio track, ffmpeg unavailable, or extraction
            # otherwise failed -- fall back to silence rather than
            # crashing the whole batch over one bad/silent video.
            num_samples = int(
                self.audio_sample_rate * self.audio_duration_seconds
            )
            audio_tensor = torch.zeros(num_samples, dtype=torch.float32)

        return {
            "frames": frames_tensor,
            "audio": audio_tensor,
            "label": label_tensor,
        }

    def close(self):
        """Release the face detector's underlying MediaPipe model."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None
