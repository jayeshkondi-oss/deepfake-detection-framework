"""
report.py
Combine the classifier verdict, temporal localization, and visual/
audio explanations into a single structured, human-readable report --
the final box in the original architecture diagram.
"""

from dataclasses import dataclass


@dataclass
class DeepfakeReport:
    """Structured result of running the full detection + explanation
    pipeline on one video."""

    video_path: str
    prediction: str  # "REAL" or "FAKE"
    fake_probability: float
    fake_probability_per_frame: list
    frame_times_seconds: list | None
    visual_saliency_available: bool
    audio_importance_available: bool
    most_suspicious_frame_index: int | None = None
    visualization_paths: dict | None = None

    def summary(self) -> str:
        """A short, human-readable text summary."""
        lines = [
            f"Video: {self.video_path}",
            f"Prediction: {self.prediction} "
            f"({self.fake_probability * 100:.1f}% fake confidence)",
        ]

        if self.most_suspicious_frame_index is not None:
            idx = self.most_suspicious_frame_index
            frame_prob = self.fake_probability_per_frame[idx]

            if self.frame_times_seconds is not None:
                where = f"~{self.frame_times_seconds[idx]:.1f}s into the clip"
            else:
                where = f"frame {idx}"

            if self.prediction == "REAL":
                lines.append(
                    f"Peak anomaly point: {where} "
                    f"({frame_prob * 100:.1f}% fake probability - within safe REAL threshold)"
                )
            else:
                lines.append(
                    f"Most suspicious moment (manipulation detected): {where} "
                    f"({frame_prob * 100:.1f}% fake confidence at that frame)"
                )


        if self.visualization_paths:
            lines.append("Saved visualizations:")
            for name, path in self.visualization_paths.items():
                lines.append(f"  - {name}: {path}")
        else:
            if self.visual_saliency_available:
                lines.append(
                    "Visual explanation available: see which face regions "
                    "the model focused on."
                )

            if self.audio_importance_available:
                lines.append(
                    "Audio explanation available: see which portions of "
                    "the audio the model focused on."
                )

        return "\n".join(lines)


def build_report(
    video_path: str,
    fake_probability: float,
    fake_probability_per_frame,
    frame_times_seconds=None,
    visual_saliency_available: bool = False,
    audio_importance_available: bool = False,
    threshold: float = 0.5,
) -> DeepfakeReport:
    """
    Args:
        video_path: path to the video that was analyzed.
        fake_probability: overall P(fake) for the whole video, in [0, 1].
        fake_probability_per_frame: iterable of per-frame P(fake) values.
        frame_times_seconds: optional iterable of timestamps matching
            fake_probability_per_frame.
        visual_saliency_available: whether visual XAI was computed.
        audio_importance_available: whether audio XAI was computed.
        threshold: confidence threshold for FAKE verdict (default 0.5).

    Returns:
        A DeepfakeReport.
    """
    per_frame = [float(p) for p in fake_probability_per_frame]

    most_suspicious_index = None
    if per_frame:
        most_suspicious_index = max(
            range(len(per_frame)), key=lambda i: per_frame[i]
        )

    return DeepfakeReport(
        video_path=str(video_path),
        prediction="FAKE" if fake_probability >= threshold else "REAL",
        fake_probability=float(fake_probability),
        fake_probability_per_frame=per_frame,
        frame_times_seconds=(
            [float(t) for t in frame_times_seconds]
            if frame_times_seconds is not None
            else None
        ),
        visual_saliency_available=visual_saliency_available,
        audio_importance_available=audio_importance_available,
        most_suspicious_frame_index=most_suspicious_index,
    )

