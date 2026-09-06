import torch

from deepfake.training.metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_confusion_matrix,
    calculate_metrics,
)


def test_accuracy():
    predictions = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 0, 0])

    accuracy = calculate_accuracy(predictions, labels)

    assert accuracy == 0.75


def test_precision_recall_f1():
    predictions = torch.tensor([1, 1, 0, 0])
    labels = torch.tensor([1, 0, 1, 0])

    precision, recall, f1 = calculate_precision_recall_f1(
        predictions,
        labels,
    )

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_confusion_matrix():
    predictions = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 0, 0])

    matrix = calculate_confusion_matrix(
        predictions,
        labels,
        num_classes=2,
    )

    expected = torch.tensor([
        [2, 1],
        [0, 1],
    ])

    assert torch.equal(matrix, expected)


def test_all_metrics():
    predictions = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 0, 0])

    metrics = calculate_metrics(
        predictions,
        labels,
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics

    assert metrics["accuracy"] == 0.75