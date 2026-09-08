from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .metrics import calculate_metrics
from .checkpoint import CheckpointManager


class Trainer:
    """
    Training engine for the deepfake detection model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        checkpoint_dir: str = "models/checkpoints",
        early_stopping_patience: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        optimizer_param_groups: Optional[list] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)

        # Use class-weighted loss if provided (fixes FAKE-bias from imbalanced data).
        # label_smoothing=0.1 prevents overconfident predictions and reduces overfitting.
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=0.1,
            )
            print(
                f"Using weighted loss - REAL weight: {class_weights[0]:.4f}, "
                f"FAKE weight: {class_weights[1]:.4f} | label_smoothing=0.1"
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Support differential learning rates via param groups.
        params = optimizer_param_groups if optimizer_param_groups else self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        self.use_amp = (
            mixed_precision
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )

        self.early_stopping_patience = early_stopping_patience

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": [],
            "train_f1": [],
            "val_f1": [],
            "learning_rate": [],
        }

        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def _move_batch_to_device(self, batch):
        """
        Move a multimodal batch to the selected device.

        Expected batch:
            frames: [B,T,3,H,W]
            audio:  [B,S]
            label:  [B]
        """

        frames = batch["frames"].to(
            self.device,
            non_blocking=True,
        )

        audio = batch.get("audio")

        if audio is not None:
            audio = audio.to(
                self.device,
                non_blocking=True,
            )

        labels = batch["label"].to(
            self.device,
            non_blocking=True,
        )

        return frames, audio, labels

    def _forward(self, frames, audio):
        """
        Forward pass supporting the multimodal detector.

        Note: DeepfakeDetector.forward() takes a parameter named
        'waveform' (not 'audio') -- we pass it through under that
        name here so this trainer works with the real model, not
        just mock models that happen to name their parameter 'audio'.
        """

        if audio is None:
            return self.model(frames)

        return self.model(
            frames=frames,
            waveform=audio,
        )

    def _extract_logits(self, output):
        """
        Support models returning either:
            Tensor
        or
            {"logits": Tensor}
        """

        if isinstance(output, dict):
            if "logits" not in output:
                raise KeyError(
                    "Model output dictionary must contain 'logits'."
                )

            return output["logits"]

        if isinstance(output, torch.Tensor):
            return output

        raise TypeError(
            "Model output must be a Tensor or dictionary containing logits."
        )

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        correct = 0
        total_samples = 0

        loader = self.train_loader
        if _TQDM_AVAILABLE:
            loader = tqdm(
                self.train_loader,
                desc="  Train",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

        for batch in loader:
            frames, audio, labels = self._move_batch_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                output = self._forward(frames, audio)
                logits = self._extract_logits(output)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.detach().item()
            total_loss += batch_loss * labels.size(0)
            total_samples += labels.size(0)

            predictions = torch.argmax(logits.detach(), dim=1)
            correct += (predictions.cpu() == labels.detach().cpu()).sum().item()

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.detach().cpu())

            if _TQDM_AVAILABLE:
                loader.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    acc=f"{correct / max(total_samples, 1):.3f}",
                )

        if len(self.train_loader.dataset) == 0:
            raise RuntimeError("Training dataset is empty.")

        epoch_loss = total_loss / len(self.train_loader.dataset)

        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        metrics = calculate_metrics(predictions, labels)

        return epoch_loss, metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in self.val_loader:
            frames, audio, labels = self._move_batch_to_device(
                batch
            )

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                output = self._forward(
                    frames,
                    audio,
                )

                logits = self._extract_logits(output)

                loss = self.criterion(
                    logits,
                    labels,
                )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

        if len(self.val_loader.dataset) == 0:
            raise RuntimeError(
                "Validation dataset is empty."
            )

        epoch_loss = (
            total_loss
            / len(self.val_loader.dataset)
        )

        predictions = torch.cat(
            all_predictions
        )

        labels = torch.cat(
            all_labels
        )

        metrics = calculate_metrics(
            predictions,
            labels,
        )

        return epoch_loss, metrics

    def fit(
        self,
        epochs: int = 30,
        resume_from: Optional[str] = None,
    ):
        """
        Train the model.

        Returns:
            Training history dictionary.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        start_epoch = 0

        if resume_from is not None:
            checkpoint = self.checkpoint_manager.load(
                path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=str(self.device),
            )

            start_epoch = (
                checkpoint["epoch"] + 1
            )

            self.best_val_loss = checkpoint.get(
                "best_val_loss",
                float("inf"),
            )

            self.history = checkpoint.get(
                "history",
                self.history,
            )

        for epoch in range(
            start_epoch,
            epochs,
        ):
            train_loss, train_metrics = (
                self.train_one_epoch()
            )

            val_loss, val_metrics = (
                self.validate()
            )

            self.scheduler.step(
                val_loss
            )

            current_lr = self.optimizer.param_groups[0][
                "lr"
            ]

            self.history["train_loss"].append(
                train_loss
            )

            self.history["val_loss"].append(
                val_loss
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            self.history["train_precision"].append(
                train_metrics["precision"]
            )

            self.history["val_precision"].append(
                val_metrics["precision"]
            )

            self.history["train_recall"].append(
                train_metrics["recall"]
            )

            self.history["val_recall"].append(
                val_metrics["recall"]
            )

            self.history["train_f1"].append(
                train_metrics["f1"]
            )

            self.history["val_f1"].append(
                val_metrics["f1"]
            )

            self.history["learning_rate"].append(
                current_lr
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                self.checkpoint_manager.save_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val_loss=val_loss,
                    history=self.history,
                )

                print(
                    "  -> New best model saved."
                )

            else:
                self.epochs_without_improvement += 1

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_val_loss=self.best_val_loss,
                history=self.history,
                filename="last_checkpoint.pt",
            )

            if (
                self.epochs_without_improvement
                >= self.early_stopping_patience
            ):
                print(
                    "Early stopping triggered."
                )
                break

        return self.history