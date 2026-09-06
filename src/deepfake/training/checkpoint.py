from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    """
    Save and load training checkpoints.

    A checkpoint contains:
        - model state
        - optimizer state
        - scheduler state
        - epoch
        - best validation loss
        - training history
    """

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
        history: dict,
        scheduler: Optional[object] = None,
        filename: str = "checkpoint.pt",
    ) -> Path:

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        path = self.checkpoint_dir / filename

        torch.save(checkpoint, path)

        return path

    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: str = "cpu",
    ) -> dict:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if scheduler is not None:
            scheduler_state = checkpoint.get(
                "scheduler_state_dict"
            )

            if scheduler_state is not None:
                scheduler.load_state_dict(
                    scheduler_state
                )

        return checkpoint

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
        history: dict,
        scheduler: Optional[object] = None,
    ) -> Path:

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
            scheduler=scheduler,
            filename="best_model.pt",
        )