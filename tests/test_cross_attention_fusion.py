import torch

from deepfake.models.cross_attention_fusion import (
    CrossAttentionFusion,
)


def test_fusion_creation():
    model = CrossAttentionFusion()

    assert model is not None
    assert model.feature_dim == 768

    print("\nCross-attention fusion created successfully.")
    print(f"Feature dimension: {model.feature_dim}")


def test_fusion_output_shape():
    model = CrossAttentionFusion()

    visual_features = torch.randn(
        2,
        16,
        768,
    )

    audio_features = torch.randn(
        2,
        20,
        768,
    )

    with torch.no_grad():
        output = model(
            visual_features,
            audio_features,
        )

    assert output.shape == (2, 768)

    print(f"\nVisual input: {visual_features.shape}")
    print(f"Audio input : {audio_features.shape}")
    print(f"Output      : {output.shape}")


def test_fusion_gpu():
    if not torch.cuda.is_available():
        print("\nCUDA unavailable - test skipped.")
        return

    device = torch.device("cuda:0")

    model = CrossAttentionFusion()
    model = model.to(device)

    visual_features = torch.randn(
        2,
        16,
        768,
        device=device,
    )

    audio_features = torch.randn(
        2,
        20,
        768,
        device=device,
    )

    with torch.no_grad():
        output = model(
            visual_features,
            audio_features,
        )

    torch.cuda.synchronize()

    assert output.device.type == "cuda"
    assert output.shape == (2, 768)

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )
    print(f"Visual device: {visual_features.device}")
    print(f"Audio device : {audio_features.device}")
    print(f"Output device: {output.device}")
    print(f"Output shape : {output.shape}")


def test_mismatched_batch_size():
    model = CrossAttentionFusion()

    visual_features = torch.randn(
        2,
        16,
        768,
    )

    audio_features = torch.randn(
        3,
        20,
        768,
    )

    try:
        model(
            visual_features,
            audio_features,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_feature_dimension():
    model = CrossAttentionFusion()

    visual_features = torch.randn(
        2,
        16,
        512,
    )

    audio_features = torch.randn(
        2,
        20,
        768,
    )

    try:
        model(
            visual_features,
            audio_features,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass