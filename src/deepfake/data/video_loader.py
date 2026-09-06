"""
video_loader.py
Read frames and metadata from a video file using OpenCV.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoLoader:
    """
    Thin wrapper around cv2.VideoCapture for reading metadata,
    single frames, or evenly-sampled sequences of frames.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))

        if not self._capture.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        self.frame_count = int(
            self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_metadata(self) -> dict:
        """Return basic video metadata."""
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }

    def read_frame(self, index: int) -> np.ndarray:
        """
        Read a single frame by index.

        Args:
            index: zero-based frame index.

        Returns:
            Frame as a numpy array [H, W, 3] (BGR, OpenCV convention).
        """
        if index < 0 or index >= self.frame_count:
            raise IndexError(
                f"Frame index {index} out of range "
                f"(video has {self.frame_count} frames)."
            )

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()

        if not ok or frame is None:
            raise IndexError(f"Could not read frame at index {index}.")

        return frame

    def sample_frames(self, num_frames: int) -> list[np.ndarray]:
        """
        Sample num_frames evenly spaced frames across the video.

        Args:
            num_frames: how many frames to sample.

        Returns:
            List of frames, each [H, W, 3].
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be greater than zero.")

        if self.frame_count <= 0:
            raise ValueError("Video has no readable frames.")

        indices = [
            min(
                int(i * self.frame_count / num_frames),
                self.frame_count - 1,
            )
            for i in range(num_frames)
        ]

        return [self.read_frame(index) for index in indices]

    def release(self):
        """Release the underlying VideoCapture."""
        self._capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
