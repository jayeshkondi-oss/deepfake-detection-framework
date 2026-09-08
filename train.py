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
from torch.utils.data import DataLoader, Subset

from deepfake.data.dataset import DeepfakeDataset
from deepfake.data.cached_dataset import CachedDeepfakeDataset
from deepfake.models.deepfake_detector import DeepfakeDetector
from deepfake.training.trainer import Trainer
from deepfake.utils.config import load_config


def build_cached_datasets(cache_dir, val_split):
    """
    Fast path: read pre-extracted .pt tensors from cache_dir.
    No face detection, no audio extraction — just tensor loads.
    Each epoch is ~20x faster than the live pipeline.
    """
    import torch

    base = CachedDeepfakeDataset(cache_dir=cache_dir, augment=False)
    augmented = CachedDeepfakeDataset(cache_dir=cache_dir, augment=True)

    n = len(base)
    val_size = max(1, int(n * val_split))
    train_size = n - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_dataset = Subset(augmented, train_idx)
    val_dataset = Subset(base, val_idx)

    print(f"Cached dataset: {n} samples ({train_size} train / {val_size} val) | augment=True on train")
    return train_dataset, val_dataset, base


def compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights so the loss treats REAL
    and FAKE equally regardless of how many of each are in the dataset.

    Returns a [2] tensor: [weight_for_REAL, weight_for_FAKE].
    """
    labels = [dataset.dataset.get_label(i) for i in dataset.indices]
    num_real = labels.count(0)
    num_fake = labels.count(1)
    total = num_real + num_fake

    print(f"Class distribution → REAL: {num_real}, FAKE: {num_fake}")

    # weight = total / (num_classes * count_for_class)
    weight_real = total / (2.0 * max(num_real, 1))
    weight_fake = total / (2.0 * max(num_fake, 1))

    return torch.tensor([weight_real, weight_fake], dtype=torch.float32)


def build_datasets(data_dir, config, val_split):
    # First, load without augmentation to count samples and compute the split.
    base_dataset = DeepfakeDataset(
        root_dir=data_dir,
        num_frames=config.data.get("sample_frames", 16),
        image_size=config.data.get("image_size", 224),
        audio_sample_rate=config.data.get("audio_sample_rate", 16000),
        augment=False,
    )

    if len(base_dataset) < 2:
        raise ValueError(
            f"Found only {len(base_dataset)} video(s) in {data_dir}. "
            "Need at least 2 (one for train, one for val) to proceed."
        )

    val_size = max(1, int(len(base_dataset) * val_split))
    train_size = len(base_dataset) - val_size

    if train_size < 1:
        raise ValueError(
            f"Only {len(base_dataset)} video(s) found -- not enough for both "
            f"a train and val split at val_split={val_split}. Add more "
            "videos or lower --val_split."
        )

    # Build an augmented version for training.
    augmented_dataset = DeepfakeDataset(
        root_dir=data_dir,
        num_frames=config.data.get("sample_frames", 16),
        image_size=config.data.get("image_size", 224),
        audio_sample_rate=config.data.get("audio_sample_rate", 16000),
        augment=True,
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Training uses the augmented dataset, validation uses the clean dataset.
    from torch.utils.data import Subset
    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)

    print(
        f"Dataset: {len(base_dataset)} videos total "
        f"({train_size} train / {val_size} val) | augment=True on train"
    )

    return train_dataset, val_dataset, base_dataset


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
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable automatic class-weight balancing (not recommended "
        "unless your dataset is perfectly balanced).",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load pre-extracted .pt tensor files from --data_dir instead of "
        "running live face detection. Run preprocess_cache.py first to "
        "generate the cache. Makes each epoch ~20x faster.",
    )
    parser.add_argument(
        "--unfreeze_vit_layers",
        type=int,
        default=0,
        help="After freezing the ViT backbone, unfreeze the last N transformer "
        "blocks so they can adapt to deepfake artifacts. These blocks use a "
        "10x lower learning rate than the classifier head. "
        "Recommended: 3. Only has effect when --freeze_visual_backbone is set.",
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

    if args.use_cache:
        print("Using cached dataset (pre-extracted tensors) -- fast mode.")
        train_dataset, val_dataset, base_dataset = build_cached_datasets(
            args.data_dir, args.val_split
        )
    else:
        train_dataset, val_dataset, base_dataset = build_datasets(
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

    # Partially unfreeze last N ViT blocks for fine-tuning.
    base_lr = config.training.get("learning_rate", 5e-5)
    backbone_lr = base_lr / 10.0   # 10x lower for backbone layers
    if args.unfreeze_vit_layers > 0 and args.freeze_visual_backbone:
        model.visual_encoder.unfreeze_last_n_layers(args.unfreeze_vit_layers)
        print(f"Differential LR -- backbone: {backbone_lr:.2e}, head: {base_lr:.2e}")
        # Build parameter groups: backbone params at lower LR.
        backbone_params = list(model.visual_encoder.vit.parameters())
        backbone_param_ids = {id(p) for p in backbone_params if p.requires_grad}
        head_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in backbone_param_ids]
        unfreeze_backbone_params = [p for p in backbone_params if p.requires_grad]
        optimizer_param_groups = [
            {"params": unfreeze_backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": base_lr},
        ]
    else:
        optimizer_param_groups = None   # Trainer uses default single LR

    epochs = args.epochs if args.epochs is not None else config.training.get("epochs", 30)

    # Compute class weights to counteract any real/fake imbalance.
    class_weights = None
    if not args.no_class_weights:
        train_labels = [base_dataset.get_label(i) for i in train_dataset.indices]
        num_real = train_labels.count(0)
        num_fake = train_labels.count(1)
        total = num_real + num_fake
        print(f"Train class distribution -> REAL: {num_real}, FAKE: {num_fake}")
        weight_real = total / (2.0 * max(num_real, 1))
        weight_fake = total / (2.0 * max(num_fake, 1))
        class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=base_lr,
        weight_decay=config.training.get("weight_decay", 1e-3),
        mixed_precision=config.training.get("mixed_precision", True),
        checkpoint_dir=config.output.get("checkpoint_dir", "models/checkpoints"),
        early_stopping_patience=config.training.get("early_stopping_patience", 5),
        class_weights=class_weights,
        optimizer_param_groups=optimizer_param_groups,
    )

    print(f"Starting training for {epochs} epoch(s)...")
    history = trainer.fit(epochs=epochs, resume_from=args.resume_from)

    print("Training complete.")
    print(f"Best val loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch + 1})")
    print(f"Checkpoints saved to: {config.output.get('checkpoint_dir', 'models/checkpoints')}")

    # Close both datasets' MediaPipe face detectors.
    base_dataset.close()


if __name__ == "__main__":
    main()
