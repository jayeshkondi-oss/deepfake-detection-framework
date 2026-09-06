import torch

from deepfake.models.visual_encoder import VisualEncoder


def test_visual_encoder_creation():
    model = VisualEncoder()

    assert model is not None
    assert model.feature_dim == 768

    print("\nVisual encoder created successfully.")
    print(f"Feature dimension: {model.feature_dim}")


def test_visual_encoder_output_shape():
    model = VisualEncoder()

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        features = model(images)

    assert features.shape == (2, 768)

    print(f"\nInput shape : {images.shape}")
    print(f"Output shape: {features.shape}")


def test_visual_encoder_gpu():
    if not torch.cuda.is_available():
        print("\nCUDA unavailable - test skipped.")
        return

    device = torch.device("cuda:0")

    model = VisualEncoder()
    model = model.to(device)

    images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        features = model(images)

    torch.cuda.synchronize()

    assert features.device.type == "cuda"
    assert features.shape == (2, 768)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Input device : {images.device}")
    print(f"Output device: {features.device}")
    print(f"Output shape : {features.shape}")


def test_visual_encoder_invalid_dimensions():
    model = VisualEncoder()

    invalid_input = torch.randn(
        2,
        224,
        224,
    )

    try:
        model(invalid_input)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_visual_encoder_invalid_channels():
    model = VisualEncoder()

    invalid_input = torch.randn(
        2,
        1,
        224,
        224,
    )

    try:
        model(invalid_input)
        assert False, "Expected ValueError"
    except ValueError:
        pass