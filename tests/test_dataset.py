from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from deepfake.data.dataset import DeepfakeDataset


def create_test_video(
    path: Path,
    num_frames: int = 8,
    width: int = 320,
    height: int = 240,
):
    """
    Create a small synthetic MP4 video for testing.
    """

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        10.0,
        (width, height),
    )

    if not writer.isOpened():
        pytest.skip(
            "OpenCV MP4 video writer is unavailable."
        )

    for i in range(num_frames):

        frame = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        # Create a simple changing pattern.
        frame[:, :, 0] = i * 20
        frame[:, :, 1] = 100
        frame[:, :, 2] = 200

        writer.write(frame)

    writer.release()


@pytest.fixture
def dataset_dir(tmp_path):
    """
    Create a temporary dataset:

        dataset/
        ├── real/
        │   └── real.mp4
        └── fake/
            └── fake.mp4
    """

    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"

    real_dir.mkdir()
    fake_dir.mkdir()

    create_test_video(
        real_dir / "real.mp4"
    )

    create_test_video(
        fake_dir / "fake.mp4"
    )

    return tmp_path


def test_dataset_creation(dataset_dir):
    dataset = DeepfakeDataset(
        dataset_dir,
        num_frames=4,
        image_size=224,
    )

    assert len(dataset) == 2


def test_dataset_labels(dataset_dir):
    dataset = DeepfakeDataset(
        dataset_dir,
        num_frames=4,
        image_size=224,
    )

    labels = [
        dataset.get_label(i)
        for i in range(len(dataset))
    ]

    assert 0 in labels
    assert 1 in labels


def test_dataset_video_paths(dataset_dir):
    dataset = DeepfakeDataset(
        dataset_dir,
        num_frames=4,
        image_size=224,
    )

    for index in range(len(dataset)):
        path = dataset.get_video_path(index)

        assert path.exists()
        assert path.suffix.lower() == ".mp4"


def test_dataset_output_shape(dataset_dir):
    dataset = DeepfakeDataset(
        dataset_dir,
        num_frames=4,
        image_size=224,
    )

    sample = dataset[0]

    assert isinstance(
        sample["frames"],
        torch.Tensor,
    )

    assert sample["frames"].shape == (
        4,
        3,
        224,
        224,
    )


def test_dataset_output_label(dataset_dir):
    dataset = DeepfakeDataset(
        dataset_dir,
        num_frames=4,
        image_size=224,
    )

    sample = dataset[0]

    assert isinstance(
        sample["label"],
        torch.Tensor,
    )

    assert sample["label"].dtype == torch.long

    assert sample["label"].item() in {
        0,
        1,
    }


def test_invalid_dataset_directory(tmp_path):
    missing_directory = (
        tmp_path / "does_not_exist"
    )

    with pytest.raises(FileNotFoundError):
        DeepfakeDataset(
            missing_directory
        )