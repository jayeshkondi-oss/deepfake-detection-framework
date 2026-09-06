import subprocess
from pathlib import Path

import pytest

from deepfake.explainability.report import DeepfakeReport
from deepfake.inference.predictor import DeepfakePredictor


@pytest.fixture(scope="module")
def predictor():
    predictor = DeepfakePredictor(pretrained=False, num_frames=4)
    yield predictor
    predictor.close()


def test_predict_returns_report(predictor, sample_video):
    report = predictor.predict(sample_video)

    assert isinstance(report, DeepfakeReport)
    assert report.prediction in ("REAL", "FAKE")
    assert 0.0 <= report.fake_probability <= 1.0
    assert len(report.fake_probability_per_frame) == 4
    assert report.visual_saliency_available
    assert report.audio_importance_available


def test_predict_without_explanations(predictor, sample_video):
    report = predictor.predict(sample_video, compute_explanations=False)

    assert not report.visual_saliency_available
    assert not report.audio_importance_available
    # Classification and localization still run either way.
    assert len(report.fake_probability_per_frame) == 4


def test_summary_is_readable(predictor, sample_video):
    report = predictor.predict(sample_video)
    summary = report.summary()

    assert report.prediction in summary
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_predict_saves_visualization_images(predictor, sample_video, tmp_path):
    report = predictor.predict(
        sample_video, save_visualizations_dir=tmp_path
    )

    assert report.visualization_paths is not None
    for name in ("visual_attention", "audio_attention", "temporal_timeline"):
        path = Path(report.visualization_paths[name])
        assert path.exists()
        assert path.stat().st_size > 0

    assert "Saved visualizations:" in report.summary()


def test_no_visualizations_saved_by_default(predictor, sample_video):
    report = predictor.predict(sample_video)

    assert report.visualization_paths is None
