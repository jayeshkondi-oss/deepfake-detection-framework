from deepfake.explainability.report import build_report


def test_fake_prediction():
    report = build_report(
        video_path="test.mp4",
        fake_probability=0.87,
        fake_probability_per_frame=[0.6, 0.9, 0.95, 0.7],
    )

    assert report.prediction == "FAKE"
    assert report.most_suspicious_frame_index == 2


def test_real_prediction():
    report = build_report(
        video_path="test.mp4",
        fake_probability=0.12,
        fake_probability_per_frame=[0.1, 0.15, 0.1],
    )

    assert report.prediction == "REAL"


def test_summary_includes_prediction_and_video_path():
    report = build_report(
        video_path="my_video.mp4",
        fake_probability=0.9,
        fake_probability_per_frame=[0.9],
    )

    summary = report.summary()

    assert "my_video.mp4" in summary
    assert "FAKE" in summary


def test_summary_with_frame_times():
    report = build_report(
        video_path="test.mp4",
        fake_probability=0.8,
        fake_probability_per_frame=[0.2, 0.9, 0.3],
        frame_times_seconds=[0.0, 0.5, 1.0],
    )

    summary = report.summary()

    assert "0.5s" in summary


def test_empty_per_frame_list_has_no_suspicious_frame():
    report = build_report(
        video_path="test.mp4",
        fake_probability=0.5,
        fake_probability_per_frame=[],
    )

    assert report.most_suspicious_frame_index is None
