"""
train.py
Entry point for training the deepfake detector end to end.

Expects a dataset directory laid out as:

    data_dir/
    ├── real/
    │   └── *.mp4
    └── fake/
        └── *.mp4

Usage:
    python train.py --data_dir data/raw

    # quick smoke test on a handful of videos, 1 epoch:
    python train.py --data_dir data/smoke_test --epochs 1
"""

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from deepfake.data.dataset import DeepfakeDataset
from deepfake.models.deepfake_detector import DeepfakeDetector
from deepfake.training.trainer import Trainer
from deepfake.utils.config import load_config


def build_datasets(data_dir, config, val_split):
    dataset = DeepfakeDataset(
        root_dir=data_dir,
        num_frames=config.data.get("sample_frames", 16),
        image_size=config.data.get("image_size", 224),
        audio_sample_rate=config.data.get("audio_sample_rate", 16000),
    )

    if len(dataset) < 2:
        raise ValueError(
            f"Found only {len(dataset)} video(s) in {data_dir}. "
            "Need at least 2 (one for train, one for val) to proceed."
        )

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    if train_size < 1:
        raise ValueError(
            f"Only {len(dataset)} video(s) found -- not enough for both "
            f"a train and val split at val_split={val_split}. Add more "
            "videos or lower --val_split."
        )

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(
        f"Dataset: {len(dataset)} videos total "
        f"({train_size} train / {val_size} val)"
    )

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing real/ and fake/ subfolders of .mp4 files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of videos held out for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from config.yaml "
        "(useful for a quick smoke test, e.g. --epochs 1).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--freeze_visual_backbone",
        action="store_true",
        help="Freeze the pretrained ViT weights, training only the "
        "smaller fusion/temporal/classifier layers on top. Recommended "
        "when you have a few hundred videos or fewer -- fully "
        "fine-tuning a huge pretrained transformer on limited data is "
        "itself a major overfitting risk.",
    )
    parser.add_argument(
        "--freeze_audio_backbone",
        action="store_true",
        help="Same as --freeze_visual_backbone, for the pretrained "
        "Wav2Vec2 weights.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print(
            "WARNING: no GPU detected. Training will be extremely slow "
            "on CPU -- this is fine for a tiny smoke test, but not for "
            "real training."
        )

    train_dataset, val_dataset = build_datasets(
        args.data_dir, config, args.val_split
    )

    batch_size = config.training.get("batch_size", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Building model (this downloads pretrained ViT + Wav2Vec2 "
          "weights on first run -- needs internet access)...")
    if args.freeze_visual_backbone or args.freeze_audio_backbone:
        print(
            f"Freezing backbones -- visual: {args.freeze_visual_backbone}, "
            f"audio: {args.freeze_audio_backbone}"
        )
    model = DeepfakeDetector(
        feature_dim=config.model.get("visual_feature_dim", 768),
        freeze_visual_backbone=args.freeze_visual_backbone,
        freeze_audio_backbone=args.freeze_audio_backbone,
    )

    epochs = args.epochs if args.epochs is not None else config.training.get("epochs", 30)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.training.get("learning_rate", 1e-4),
        weight_decay=config.training.get("weight_decay", 1e-4),
        mixed_precision=config.training.get("mixed_precision", True),
        checkpoint_dir=config.output.get("checkpoint_dir", "models/checkpoints"),
        early_stopping_patience=config.training.get("early_stopping_patience", 5),
    )

    print(f"Starting training for {epochs} epoch(s)...")
    history = trainer.fit(epochs=epochs, resume_from=args.resume_from)

    print("Training complete.")
    print(f"Best val loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch + 1})")
    print(f"Checkpoints saved to: {config.output.get('checkpoint_dir', 'models/checkpoints')}")

    train_dataset.dataset.close()


if __name__ == "__main__":
    main()
