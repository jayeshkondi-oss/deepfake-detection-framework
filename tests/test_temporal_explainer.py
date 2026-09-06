import pytest
import torch

from deepfake.explainability.temporal_explainer import TemporalExplainer
from deepfake.models.deepfake_detector import DeepfakeDetector


@pytest.fixture(scope="module")
def explainer():
    model = DeepfakeDetector(pretrained=False)
    return TemporalExplainer(model)


def test_output_shape(explainer):
    frames = torch.randn(6, 3, 224, 224)
    waveform = torch.randn(16000)

    result = explainer.explain(frames, waveform)

    assert result["fake_probability_per_frame"].shape == (6,)


def test_probabilities_in_valid_range(explainer):
    frames = torch.randn(4, 3, 224, 224)
    waveform = torch.randn(16000)

    result = explainer.explain(frames, waveform)
    probs = result["fake_probability_per_frame"]

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_with_fps_gives_timestamps(explainer):
    frames = torch.randn(4, 3, 224, 224)
    waveform = torch.randn(16000)

    result = explainer.explain(frames, waveform, fps=2.0)

    assert "times_seconds" in result
    assert torch.allclose(
        result["times_seconds"], torch.tensor([0.0, 0.5, 1.0, 1.5])
    )


def test_without_fps_omits_timestamps(explainer):
    frames = torch.randn(4, 3, 224, 224)
    waveform = torch.randn(16000)

    result = explainer.explain(frames, waveform)

    assert "times_seconds" not in result


def test_invalid_frames_shape(explainer):
    with pytest.raises(ValueError):
        explainer.explain(torch.randn(3, 224, 224), torch.randn(16000))


def test_invalid_waveform_shape(explainer):
    with pytest.raises(ValueError):
        explainer.explain(torch.randn(4, 3, 224, 224), torch.randn(1, 16000))
