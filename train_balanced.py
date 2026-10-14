"""
train_balanced.py
Trains the multi-modal fusion, temporal transformer, and classification head
with frozen ViT & Wav2Vec2 backbones on a perfectly balanced dataset:
- 477 REAL videos
- 477 FAKE videos (238 DFDC fakes + 239 TalkingHeadBench modern AI fakes)
Total: 954 videos.

Prevents overfitting and false positives on real videos while adapting the head
to detect modern generative AI and classic deepfakes.
"""

import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from deepfake.data.cached_dataset import CachedDeepfakeDataset
from deepfake.models.deepfake_detector import DeepfakeDetector
from deepfake.training.metrics import calculate_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load clean and augmented cached datasets
    clean_ds = CachedDeepfakeDataset("data/combined_cache", augment=False)
    aug_ds = CachedDeepfakeDataset("data/combined_cache", augment=True)

    # Separate real and fake indices
    real_indices = [i for i, (p, label) in enumerate(clean_ds.samples) if label == 0]
    thb_indices = [i for i, (p, label) in enumerate(clean_ds.samples) if label == 1 and "thb_" in p.name]
    dfdc_indices = [i for i, (p, label) in enumerate(clean_ds.samples) if label == 1 and "thb_" not in p.name]

    print(f"Available: {len(real_indices)} REAL, {len(dfdc_indices)} DFDC FAKE, {len(thb_indices)} THB FAKE")

    # Balanced selection: 477 Real + 238 DFDC + 239 THB = 954 samples
    random.seed(42)
    torch.manual_seed(42)

    selected_dfdc = random.sample(dfdc_indices, min(238, len(dfdc_indices)))
    selected_thb = random.sample(thb_indices, min(239, len(thb_indices)))
    selected_fake = selected_dfdc + selected_thb

    all_selected = [(idx, 0) for idx in real_indices] + [(idx, 1) for idx in selected_fake]
    random.shuffle(all_selected)

    # 85% train, 15% val split
    n_total = len(all_selected)
    n_val = int(0.15 * n_total)
    n_train = n_total - n_val

    train_pairs = all_selected[:n_train]
    val_pairs = all_selected[n_train:]

    train_indices = [p[0] for p in train_pairs]
    val_indices = [p[0] for p in val_pairs]

    n_train_real = sum(1 for p in train_pairs if p[1] == 0)
    n_train_fake = sum(1 for p in train_pairs if p[1] == 1)
    n_val_real = sum(1 for p in val_pairs if p[1] == 0)
    n_val_fake = sum(1 for p in val_pairs if p[1] == 1)

    print(f"Dataset split: {n_train} Train (Real: {n_train_real}, Fake: {n_train_fake}) | "
          f"{n_val} Val (Real: {n_val_real}, Fake: {n_val_fake})")

    train_loader = DataLoader(Subset(aug_ds, train_indices), batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(clean_ds, val_indices), batch_size=8, shuffle=False, num_workers=0)

    # Load baseline model with frozen backbones
    model = DeepfakeDetector(
        feature_dim=768,
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )
    base_ckpt = "models/checkpoints/best_model_backup_pre_finetune.pt"
    print(f"Loading base weights from: {base_ckpt}")
    ckpt_data = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state)
    model.to(device)

    # Parameters to train: only temporal transformer, fusion, classifier
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-3)
    # Give slight extra weight to REAL class (1.25) to strictly deter false positives
    class_weights = torch.tensor([1.25, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    epochs = 5
    best_val_loss = float("inf")
    best_val_acc = 0.0

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting balanced training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            frames = batch["frames"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(frames=frames, waveform=audio)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_labels_list = []
        val_probs_list = []
        val_real_correct = 0
        val_real_total = 0

        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device, non_blocking=True)
                audio = batch["audio"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                    logits = model(frames=frames, waveform=audio)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = logits.argmax(dim=-1)

                val_preds_list.extend(preds.cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())
                val_probs_list.extend(probs.cpu().tolist())

                for p, l in zip(preds, labels):
                    if l.item() == 0:
                        val_real_total += 1
                        if p.item() == 0:
                            val_real_correct += 1

        epoch_val_loss = val_loss / max(len(val_labels_list), 1)
        val_preds_t = torch.tensor(val_preds_list)
        val_labels_t = torch.tensor(val_labels_list)
        metrics = calculate_metrics(val_preds_t, val_labels_t)
        epoch_val_acc = metrics["accuracy"] * 100.0
        val_real_acc = val_real_correct / max(val_real_total, 1) * 100.0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.1f}%, "
            f"Real-Acc: {val_real_acc:.1f}%, F1: {metrics['f1']:.4f}"
        )

        if epoch_val_loss < best_val_loss or (val_real_acc >= 90.0 and epoch_val_acc > best_val_acc):
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            save_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "val_real_accuracy": val_real_acc,
            }, save_path)
            print(f"  -> Saved new best model to {save_path} (Val Loss: {epoch_val_loss:.4f})")

    print(f"\nTraining finished! Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.1f}%")

if __name__ == "__main__":
    main()
