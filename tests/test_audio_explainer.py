import pytest
import torch

from deepfake.explainability.audio_explainer import AudioExplainer
from deepfake.models.audio_encoder import AudioEncoder


@pytest.fixture(scope="module")
def explainer():
    encoder = AudioEncoder(pretrained=False)
    return AudioExplainer(encoder)


def test_single_waveform_output(explainer):
    waveform = torch.randn(16000)
    result = explainer.explain(waveform)

    assert "importance" in result
    assert result["importance"].ndim == 1


def test_values_normalized_to_unit_range(explainer):
    waveform = torch.randn(16000)
    result = explainer.explain(waveform)

    assert result["importance"].min() >= 0.0
    assert result["importance"].max() <= 1.0 + 1e-5


def test_with_duration_seconds(explainer):
    waveform = torch.randn(16000)
    result = explainer.explain(waveform, duration_seconds=1.0)

    assert "times_seconds" in result
    assert result["times_seconds"].shape == result["importance"].shape


def test_without_duration_seconds_omits_times(explainer):
    waveform = torch.randn(16000)
    result = explainer.explain(waveform)

    assert "times_seconds" not in result


def test_invalid_input_type(explainer):
    with pytest.raises(TypeError):
        explainer.explain([1, 2, 3])


def test_invalid_shape(explainer):
    with pytest.raises(ValueError):
        explainer.explain(torch.randn(2, 3, 16000))
