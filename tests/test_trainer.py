import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deepfake.training.trainer import Trainer


class TinyDeepfakeModel(nn.Module):
    """
    Small model used only to test the training engine.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, frames, audio=None):
        # frames: [B, 6]
        return self.network(frames)


def create_loader():
    torch.manual_seed(42)

    features = torch.tensor(
        [
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [-1., 0., 0., 0., 0., 0.],
            [0., -1., 0., 0., 0., 0.],
            [-1., -1., 0., 0., 0., 0.],
            [0., 0., -1., 0., 0., 0.],
        ]
    )

    labels = torch.tensor(
        [1, 1, 1, 0, 0, 0, 0, 0]
    )

    dataset = TensorDataset(
        features,
        labels,
    )

    class WrapperDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            features, label = dataset[index]

            return {
                "frames": features,
                "label": label,
            }

    return DataLoader(
        WrapperDataset(),
        batch_size=4,
        shuffle=True,
    )


def test_trainer_initialization(tmp_path):
    model = TinyDeepfakeModel()
    loader = create_loader()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        weight_decay=0.0001,
        checkpoint_dir=str(tmp_path),
        mixed_precision=True,
    )

    assert trainer.model is model
    assert trainer.criterion is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None

    if torch.cuda.is_available():
        assert trainer.device.type == "cuda"
    else:
        assert trainer.device.type == "cpu"


def test_train_one_epoch(tmp_path):
    model = TinyDeepfakeModel()
    loader = create_loader()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path),
    )

    loss, metrics = trainer.train_one_epoch()

    assert loss >= 0
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_validation(tmp_path):
    model = TinyDeepfakeModel()
    loader = create_loader()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path),
    )

    loss, metrics = trainer.validate()

    assert loss >= 0
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_fit_and_checkpoint(tmp_path):
    model = TinyDeepfakeModel()
    loader = create_loader()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path),
        early_stopping_patience=5,
    )

    history = trainer.fit(epochs=3)

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert len(history["train_accuracy"]) == 3
    assert len(history["val_accuracy"]) == 3

    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "last_checkpoint.pt").exists()


def test_training_updates_model(tmp_path):
    torch.manual_seed(123)

    model = TinyDeepfakeModel()
    loader = create_loader()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        learning_rate=0.01,
        checkpoint_dir=str(tmp_path),
    )

    # Store the original parameters on CPU.
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.train_one_epoch()

    changed = False

    for name, parameter in trainer.model.named_parameters():
        if not torch.equal(
            before[name],
            parameter.detach().cpu(),
        ):
            changed = True
            break

    assert changed