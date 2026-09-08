"""
preprocess_cache.py
Run this ONCE before training. Pre-extracts face-cropped frames and
audio tensors for every video and saves them as .pt files.

Subsequent training reads .pt files instantly from disk instead of
running face detection every epoch -- makes each epoch ~20x faster.

Usage:
    .venv\\Scripts\\python.exe preprocess_cache.py --data_dir data\\combined
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake.data.audio_window import AudioWindowExtractor
from deepfake.data.face_detector import FaceDetector
from deepfake.data.preprocessing import FramePreprocessor
from deepfake.data.video_loader import VideoLoader

NUM_FRAMES = 16
IMAGE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0
FACE_MARGIN = 0.2


def process_video(video_path, face_detector, preprocessor, audio_extractor):
    with VideoLoader(video_path) as loader:
        raw_frames = loader.sample_frames(NUM_FRAMES)

    face_frames = []
    for frame in raw_frames:
        rgb_frame = frame[:, :, ::-1]
        detections = face_detector.detect(rgb_frame)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            face_frames.append(
                face_detector.crop_face(frame, best, margin=FACE_MARGIN)
            )
        else:
            face_frames.append(frame)

    frames_tensor = preprocessor.process_frames(face_frames)

    try:
        audio_tensor = audio_extractor.extract(video_path)
    except Exception:
        num_samples = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
        audio_tensor = torch.zeros(num_samples, dtype=torch.float32)

    return frames_tensor, audio_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Root dir with real/ and fake/ subfolders of .mp4 files."
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="Where to save cache (default: <data_dir>_cache alongside data_dir)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if cache file already exists."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = data_dir.parent / (data_dir.name + "_cache")

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source : {data_dir}")
    print(f"Cache  : {cache_dir}")

    face_detector = FaceDetector()
    preprocessor = FramePreprocessor(image_size=IMAGE_SIZE)
    audio_extractor = AudioWindowExtractor(
        sample_rate=AUDIO_SAMPLE_RATE,
        duration_seconds=AUDIO_DURATION,
    )

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for label in ["real", "fake"]:
        src_dir = data_dir / label
        if not src_dir.exists():
            print(f"  Skipping missing folder: {src_dir}")
            continue

        dst_dir = cache_dir / label
        dst_dir.mkdir(exist_ok=True)

        videos = sorted(src_dir.glob("*.mp4"))
        print(f"\n[{label.upper()}] {len(videos)} videos -> {dst_dir}")

        for i, video_path in enumerate(videos):
            cache_path = dst_dir / (video_path.stem + ".pt")

            if cache_path.exists() and not args.force:
                total_skipped += 1
                continue

            try:
                frames, audio = process_video(
                    video_path, face_detector, preprocessor, audio_extractor
                )
                torch.save({"frames": frames, "audio": audio}, cache_path)
                total_processed += 1
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1:4d}/{len(videos)}] {video_path.name}")
            except Exception as e:
                total_errors += 1
                print(f"  [{i+1:4d}/{len(videos)}] ERROR {video_path.name}: {e}")

    face_detector.close()

    real_count = len(list((cache_dir / "real").glob("*.pt")))
    fake_count = len(list((cache_dir / "fake").glob("*.pt")))

    print(f"\n=== Preprocessing complete ===")
    print(f"Processed : {total_processed}")
    print(f"Skipped   : {total_skipped} (already cached)")
    print(f"Errors    : {total_errors}")
    print(f"Cache     : {real_count} real + {fake_count} fake = {real_count+fake_count} total .pt files")
    print(f"\nNow run training with the cache (much faster):")
    print(f"  .venv\\Scripts\\python.exe train.py --data_dir {cache_dir} --use_cache --epochs 30 --freeze_visual_backbone --freeze_audio_backbone")


if __name__ == "__main__":
    main()
