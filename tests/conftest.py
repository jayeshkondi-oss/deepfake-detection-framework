"""
conftest.py
Shared pytest fixtures.
"""

import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """
    A tiny synthetic .mp4 (2 seconds, test pattern + a 440Hz tone),
    generated fresh by ffmpeg so tests never depend on a real video
    file being manually placed on disk at a fixed path. Automatically
    skipped if ffmpeg isn't available.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available on PATH.")

    video_dir = tmp_path_factory.mktemp("video_fixtures")
    video_path = video_dir / "sample_video.mp4"

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"Could not generate sample video with ffmpeg: {result.stderr}")

    return video_path
