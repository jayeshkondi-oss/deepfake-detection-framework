import pytest
import torch

from deepfake.explainability.visual_explainer import VisualExplainer
from deepfake.models.visual_encoder import VisualEncoder


@pytest.fixture(scope="module")
def explainer():
    encoder = VisualEncoder(pretrained=False)
    return VisualExplainer(encoder)


def test_single_frame_shape(explainer):
    frame = torch.randn(3, 224, 224)
    saliency = explainer.explain(frame)

    assert saliency.shape == (explainer.grid_size, explainer.grid_size)


def test_batch_of_frames_shape(explainer):
    frames = torch.randn(5, 3, 224, 224)
    saliency = explainer.explain(frames)

    assert saliency.shape == (5, explainer.grid_size, explainer.grid_size)


def test_values_normalized_to_unit_range(explainer):
    frames = torch.randn(3, 3, 224, 224)
    saliency = explainer.explain(frames)

    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0 + 1e-5


def test_invalid_input_type(explainer):
    with pytest.raises(TypeError):
        explainer.explain([1, 2, 3])


def test_invalid_shape(explainer):
    with pytest.raises(ValueError):
        explainer.explain(torch.randn(3, 224))
