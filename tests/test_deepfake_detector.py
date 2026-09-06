import pytest
import torch

from deepfake.models.deepfake_detector import DeepfakeDetector


def test_deepfake_detector_creation():
    model = DeepfakeDetector(
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )

    assert model.feature_dim == 768
    assert model.num_classes == 2

    print("\nComplete Deepfake Detector created successfully.")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Classes          : {model.num_classes}")


def test_deepfake_detector_output_shape():
    model = DeepfakeDetector(
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )

    model.eval()

    batch_size = 1
    num_frames = 4
    image_size = 224

    frames = torch.randn(
        batch_size,
        num_frames,
        3,
        image_size,
        image_size,
    )

    # Short waveform for integration testing.
    waveform = torch.randn(
        batch_size,
        16000,
    )

    with torch.no_grad():
        output = model(
            frames,
            waveform,
        )

    assert output.shape == (
        batch_size,
        2,
    )

    print("\nInput frames :", frames.shape)
    print("Input audio  :", waveform.shape)
    print("Output logits:", output.shape)


def test_deepfake_detector_probabilities():
    model = DeepfakeDetector(
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )

    model.eval()

    frames = torch.randn(
        1,
        2,
        3,
        224,
        224,
    )

    waveform = torch.randn(
        1,
        16000,
    )

    with torch.no_grad():
        probabilities = model.predict_proba(
            frames,
            waveform,
        )

    assert probabilities.shape == (1, 2)

    # Probabilities must be between 0 and 1.
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    # REAL + FAKE probabilities must equal 1.
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(1),
        atol=1e-5,
    )

    print("\nProbabilities:", probabilities)
    print(
        "Probability sum:",
        probabilities.sum(dim=1),
    )


def test_deepfake_detector_prediction():
    model = DeepfakeDetector(
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )

    model.eval()

    frames = torch.randn(
        1,
        2,
        3,
        224,
        224,
    )

    waveform = torch.randn(
        1,
        16000,
    )

    with torch.no_grad():
        prediction = model.predict(
            frames,
            waveform,
        )

    assert prediction.shape == (1,)

    assert prediction.item() in (0, 1)

    print("\nPrediction:", prediction.item())

    if prediction.item() == 0:
        print("Predicted class: REAL")
    else:
        print("Predicted class: FAKE")


def test_deepfake_detector_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    device = torch.device("cuda:0")

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    model = DeepfakeDetector(
        freeze_visual_backbone=True,
        freeze_audio_backbone=True,
    )

    model = model.to(device)
    model.eval()

    frames = torch.randn(
        1,
        2,
        3,
        224,
        224,
        device=device,
    )

    waveform = torch.randn(
        1,
        16000,
        device=device,
    )

    with torch.no_grad():
        output = model(
            frames,
            waveform,
        )

    assert output.device.type == "cuda"
    assert output.shape == (1, 2)

    print("Frames device :", frames.device)
    print("Audio device  :", waveform.device)
    print("Output device :", output.device)
    print("Output shape  :", output.shape)