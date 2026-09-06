import pytest
import torch

from deepfake.models.classification_head import ClassificationHead


def test_classification_head_creation():
    model = ClassificationHead()

    assert model.input_dim == 768
    assert model.hidden_dim == 384
    assert model.num_classes == 2

    print("\nClassification head created successfully.")
    print(f"Input dimension : {model.input_dim}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Classes         : {model.num_classes}")


def test_classification_head_output_shape():
    model = ClassificationHead()

    batch_size = 4

    features = torch.randn(
        batch_size,
        768,
    )

    output = model(features)

    assert output.shape == (
        batch_size,
        2,
    )

    print("\nInput shape :", features.shape)
    print("Output shape:", output.shape)


def test_classification_head_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = torch.device("cuda:0")

    model = ClassificationHead().to(device)

    features = torch.randn(
        4,
        768,
        device=device,
    )

    output = model(features)

    assert output.device.type == "cuda"
    assert output.shape == (4, 2)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("Input device :", features.device)
    print("Output device:", output.device)
    print("Output shape :", output.shape)


def test_invalid_feature_dimension():
    model = ClassificationHead()

    features = torch.randn(
        4,
        512,
    )

    with pytest.raises(ValueError):
        model(features)


def test_invalid_input_dimensions():
    model = ClassificationHead()

    features = torch.randn(
        768,
    )

    with pytest.raises(ValueError):
        model(features)