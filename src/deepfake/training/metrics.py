import torch


def calculate_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate classification accuracy.

    predictions: [N] predicted class indices
    labels:      [N] ground-truth class indices
    """
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape.")

    if labels.numel() == 0:
        raise ValueError("labels cannot be empty.")

    correct = (predictions == labels).sum().item()
    total = labels.numel()

    return correct / total


def calculate_precision_recall_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    positive_class: int = 1,
) -> tuple[float, float, float]:
    """
    Calculate precision, recall and F1 for the positive class.

    Class convention:
        0 = real
        1 = fake
    """
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape.")

    if labels.numel() == 0:
        raise ValueError("labels cannot be empty.")

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    true_positive = (
        (predictions == positive_class)
        & (labels == positive_class)
    ).sum().item()

    false_positive = (
        (predictions == positive_class)
        & (labels != positive_class)
    ).sum().item()

    false_negative = (
        (predictions != positive_class)
        & (labels == positive_class)
    ).sum().item()

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return precision, recall, f1


def calculate_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Calculate a multiclass confusion matrix.

    Rows    = actual labels
    Columns = predicted labels
    """
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape.")

    if labels.numel() == 0:
        raise ValueError("labels cannot be empty.")

    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    predictions = predictions.reshape(-1).long()
    labels = labels.reshape(-1).long()

    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    for actual, predicted in zip(labels, predictions):
        actual_index = int(actual.item())
        predicted_index = int(predicted.item())

        if not 0 <= actual_index < num_classes:
            raise ValueError("Label contains an invalid class index.")

        if not 0 <= predicted_index < num_classes:
            raise ValueError("Prediction contains an invalid class index.")

        matrix[actual_index, predicted_index] += 1

    return matrix


def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Calculate all basic classification metrics.
    """
    accuracy = calculate_accuracy(predictions, labels)

    precision, recall, f1 = calculate_precision_recall_f1(
        predictions,
        labels,
    )

    confusion_matrix = calculate_confusion_matrix(
        predictions,
        labels,
        num_classes=2,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix,
    }