import torch
import torch.nn as nn

from deepfake.training.checkpoint import CheckpointManager


def test_checkpoint_save_and_load(tmp_path):
    model = nn.Linear(10, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    manager = CheckpointManager(
        checkpoint_dir=tmp_path
    )

    history = {
        "train_loss": [0.8, 0.5],
        "val_loss": [0.7, 0.4],
    }

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=2,
        best_val_loss=0.4,
        history=history,
        filename="test_checkpoint.pt",
    )

    assert checkpoint_path.exists()

    new_model = nn.Linear(10, 2)

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=0.001,
    )

    checkpoint = manager.load(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device="cpu",
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["best_val_loss"] == 0.4
    assert checkpoint["history"] == history

    for original, restored in zip(
        model.parameters(),
        new_model.parameters(),
    ):
        assert torch.equal(original, restored)


def test_save_best(tmp_path):
    model = nn.Linear(10, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    manager = CheckpointManager(
        checkpoint_dir=tmp_path
    )

    path = manager.save_best(
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_val_loss=0.25,
        history={
            "train_loss": [0.5],
            "val_loss": [0.25],
        },
    )

    assert path.exists()
    assert path.name == "best_model.pt"