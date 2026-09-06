import cv2
import numpy as np
import pytest

from deepfake.data.video_loader import VideoLoader


@pytest.fixture
def sample_video(tmp_path):
    video_path = tmp_path / "sample.mp4"

    width = 224
    height = 224
    fps = 10
    frame_count = 20

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for i in range(frame_count):
        frame = np.full(
            (height, width, 3),
            i * 10,
            dtype=np.uint8,
        )

        writer.write(frame)

    writer.release()

    return video_path


def test_video_metadata(sample_video):
    loader = VideoLoader(sample_video)

    metadata = loader.get_metadata()

    assert metadata["frame_count"] == 20
    assert metadata["fps"] == 10
    assert metadata["width"] == 224
    assert metadata["height"] == 224


def test_read_frame(sample_video):
    loader = VideoLoader(sample_video)

    frame = loader.read_frame(0)

    assert frame is not None
    assert frame.shape == (224, 224, 3)


def test_sample_frames(sample_video):
    loader = VideoLoader(sample_video)

    frames = loader.sample_frames(16)

    assert len(frames) == 16

    for frame in frames:
        assert frame.shape == (224, 224, 3)


def test_invalid_frame_index(sample_video):
    loader = VideoLoader(sample_video)

    with pytest.raises(IndexError):
        loader.read_frame(100)