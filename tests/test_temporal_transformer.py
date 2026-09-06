import torch

from deepfake.models.temporal_transformer import (
    TemporalTransformer,
)


def test_temporal_transformer_creation():
    model = TemporalTransformer()

    assert model is not None
    assert model.feature_dim == 768
    assert model.max_sequence_length == 16

    print("\nTemporal Transformer created successfully.")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Maximum sequence : {model.max_sequence_length}")


def test_temporal_transformer_output_shape():
    model = TemporalTransformer()

    features = torch.randn(
        2,
        16,
        768,
    )

    with torch.no_grad():
        output = model(features)

    assert output.shape == (2, 768)

    print(f"\nInput shape : {features.shape}")
    print(f"Output shape: {output.shape}")


def test_temporal_transformer_gpu():
    if not torch.cuda.is_available():
        print("\nCUDA unavailable - test skipped.")
        return

    device = torch.device("cuda:0")

    model = TemporalTransformer()
    model = model.to(device)

    features = torch.randn(
        2,
        16,
        768,
        device=device,
    )

    with torch.no_grad():
        output = model(features)

    torch.cuda.synchronize()

    assert output.device.type == "cuda"
    assert output.shape == (2, 768)

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )
    print(f"Input device : {features.device}")
    print(f"Output device: {output.device}")
    print(f"Output shape : {output.shape}")


def test_invalid_feature_dimension():
    model = TemporalTransformer()

    features = torch.randn(
        2,
        16,
        512,
    )

    try:
        model(features)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_sequence_too_long():
    model = TemporalTransformer(
        max_sequence_length=16
    )

    features = torch.randn(
        2,
        20,
        768,
    )

    try:
        model(features)
        assert False, "Expected ValueError"
    except ValueError:
        pass