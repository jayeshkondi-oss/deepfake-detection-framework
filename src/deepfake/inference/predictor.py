"""
predictor.py
End-to-end inference: load a trained checkpoint, run a video through
preprocessing -> model -> classification -> temporal localization ->
visual/audio explanations -> human-readable report. This is the
"INFERENCE" + "XAI MODULE" + "HUMAN-READABLE REPORT" boxes from the
original architecture diagram, wired together into one call.
"""

from pathlib import Path

import numpy as np
import torch

from deepfake.data.audio_window import AudioWindowExtractor
from deepfake.data.face_detector import FaceDetector
from deepfake.data.preprocessing import FramePreprocessor
from deepfake.data.video_loader import VideoLoader
from deepfake.explainability.audio_explainer import AudioExplainer
from deepfake.explainability.forensic_analyzer import ForensicAnalyzer
from deepfake.explainability.report import DeepfakeReport, build_report
from deepfake.explainability.temporal_explainer import TemporalExplainer
from deepfake.explainability.visual_explainer import VisualExplainer
from deepfake.explainability.visualize import (
    save_audio_importance_plot,
    save_temporal_timeline_plot,
    save_visual_saliency,
)
from deepfake.models.deepfake_detector import DeepfakeDetector
from deepfake.training.checkpoint import CheckpointManager


class DeepfakePredictor:
    """
    Loads a trained DeepfakeDetector and runs the complete pipeline on a single video file.
    Includes temporal face-tracking interpolation, test-time augmentation (TTA),
    and comprehensive multi-modal forensic diagnostics.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        num_frames: int = 16,
        image_size: int = 224,
        face_margin: float = 0.2,
        audio_sample_rate: int = 16000,
        audio_duration_seconds: float = 4.0,
        pretrained: bool = True,
        fake_threshold: float = 0.65,
    ):
        self.device = device
        self.num_frames = num_frames
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration_seconds = audio_duration_seconds
        self.fake_threshold = fake_threshold

        self.model = DeepfakeDetector(pretrained=pretrained).to(device)

        if checkpoint_path is not None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(Path(checkpoint_path).parent)
            )
            checkpoint_manager.load(
                path=str(checkpoint_path),
                model=self.model,
                device=device,
            )

        self.model.eval()

        # Sensitive face detector for in-the-wild / unseen videos
        self.face_detector = FaceDetector(min_confidence=0.25)
        self.frame_preprocessor = FramePreprocessor(image_size=image_size)
        self.face_margin = face_margin
        self.audio_extractor = AudioWindowExtractor(
            sample_rate=audio_sample_rate,
            duration_seconds=audio_duration_seconds,
        )

        self.visual_explainer = VisualExplainer(self.model.visual_encoder)
        self.audio_explainer = AudioExplainer(self.model.audio_encoder)
        self.temporal_explainer = TemporalExplainer(self.model)
        self.forensic_analyzer = ForensicAnalyzer(grid_size=14)

    def _load_frames_and_audio(self, video_path: Path):
        with VideoLoader(video_path) as loader:
            raw_frames = loader.sample_frames(self.num_frames)
            metadata = loader.get_metadata()

        # Collect detections across all frames
        indexed_dets = []
        for i, frame in enumerate(raw_frames):
            rgb_frame = frame[:, :, ::-1]
            detections = self.face_detector.detect(rgb_frame)
            if detections:
                best = max(detections, key=lambda d: d.confidence)
                indexed_dets.append((i, best.bbox))

        face_crops = []
        h, w = raw_frames[0].shape[:2]

        # Robust Temporal Bounding Box Interpolation & Tracking
        if indexed_dets:
            known_indices = [d[0] for d in indexed_dets]
            known_boxes = np.array([d[1] for d in indexed_dets], dtype=float)
            all_boxes = []

            for i in range(len(raw_frames)):
                if i in known_indices:
                    idx = known_indices.index(i)
                    all_boxes.append(tuple(map(int, known_boxes[idx])))
                elif i < known_indices[0]:
                    all_boxes.append(tuple(map(int, known_boxes[0])))
                elif i > known_indices[-1]:
                    all_boxes.append(tuple(map(int, known_boxes[-1])))
                else:
                    b_idx = max(j for j in known_indices if j < i)
                    a_idx = min(j for j in known_indices if j > i)
                    b_box = known_boxes[known_indices.index(b_idx)]
                    a_box = known_boxes[known_indices.index(a_idx)]
                    alpha = (i - b_idx) / (a_idx - b_idx)
                    interp = (1 - alpha) * b_box + alpha * a_box
                    all_boxes.append(tuple(map(int, interp)))

            for i, frame in enumerate(raw_frames):
                x1, y1, x2, y2 = all_boxes[i]
                bw = x2 - x1
                bh = y2 - y1
                bx1 = max(0, int(x1 - bw * self.face_margin))
                by1 = max(0, int(y1 - bh * self.face_margin))
                bx2 = min(w, int(x2 + bw * self.face_margin))
                by2 = min(h, int(y2 + bh * self.face_margin))
                crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    crop = frame
                face_crops.append(crop)
        else:
            # Fallback: portrait center crop prior (upper 65% height, center 70% width)
            for frame in raw_frames:
                cy1 = int(h * 0.10)
                cy2 = int(h * 0.75)
                cx1 = int(w * 0.15)
                cx2 = int(w * 0.85)
                face_crops.append(frame[cy1:cy2, cx1:cx2])

        frames_tensor = self.frame_preprocessor.process_frames(face_crops)

        try:
            audio_tensor = self.audio_extractor.extract(video_path)
        except Exception:
            num_samples = int(
                self.audio_sample_rate * self.audio_duration_seconds
            )
            audio_tensor = torch.zeros(num_samples, dtype=torch.float32)

        sampled_fps = None
        if metadata["fps"] > 0 and metadata["frame_count"] > 0:
            video_duration = metadata["frame_count"] / metadata["fps"]
            if video_duration > 0:
                sampled_fps = self.num_frames / video_duration

        return frames_tensor, audio_tensor, sampled_fps, face_crops


    @torch.no_grad()
    def predict(
        self,
        video_path: str | Path,
        compute_explanations: bool = True,
        save_visualizations_dir: str | Path | None = None,
    ) -> DeepfakeReport:
        """
        Run the full pipeline on one video.

        Args:
            video_path: path to a video file.
            compute_explanations: if False, skip the (slower) visual
                and audio attention-rollout explanations and only
                compute the classification + temporal localization.
            save_visualizations_dir: if given (and compute_explanations
                is True), save three PNGs here: a visual attention
                heatmap over the most suspicious frame, an audio
                attention-over-time plot, and the per-frame fake-
                probability timeline.

        Returns:
            A DeepfakeReport. If images were saved, their paths are
            in report.visualization_paths.
        """
        video_path = Path(video_path)

        frames_tensor, audio_tensor, sampled_fps, face_crops = (
            self._load_frames_and_audio(video_path)
        )

        frames_tensor = frames_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)

        # Test-Time Augmentation (TTA): average predictions from original and
        # horizontally flipped frames to improve generalization on unseen/out-of-distribution videos
        probs_orig = self.model.predict_proba(
            frames_tensor.unsqueeze(0),
            audio_tensor.unsqueeze(0),
        )
        flipped_frames = torch.flip(frames_tensor, dims=[-1])
        probs_flip = self.model.predict_proba(
            flipped_frames.unsqueeze(0),
            audio_tensor.unsqueeze(0),
        )
        probabilities = 0.5 * (probs_orig + probs_flip)
        fake_probability = probabilities[0, 1].item()

        # Apply confidence threshold: must be >= fake_threshold to be FAKE.
        is_fake = fake_probability >= self.fake_threshold
        print(
            f"Fake probability: {fake_probability:.4f} | "
            f"Threshold: {self.fake_threshold:.2f} | "
            f"Verdict: {'FAKE' if is_fake else 'REAL'}"
        )

        temporal_result = self.temporal_explainer.explain(
            frames_tensor, audio_tensor, fps=sampled_fps
        )
        per_frame_probs = temporal_result["fake_probability_per_frame"]

        visual_saliency_available = False
        audio_importance_available = False
        visualization_paths = None
        visual_saliency = None

        if compute_explanations:
            visual_saliency = self.visual_explainer.explain(frames_tensor)
            visual_saliency_available = True

            audio_result = self.audio_explainer.explain(
                audio_tensor, duration_seconds=self.audio_duration_seconds
            )
            audio_importance_available = True

            if save_visualizations_dir is not None:
                save_dir = Path(save_visualizations_dir)
                most_suspicious_idx = int(per_frame_probs.argmax().item())

                visualization_paths = {
                    "visual_attention": str(
                        save_visual_saliency(
                            frames_tensor[most_suspicious_idx],
                            visual_saliency[most_suspicious_idx],
                            save_dir / "visual_attention.png",
                        )
                    ),
                    "audio_attention": str(
                        save_audio_importance_plot(
                            audio_result["importance"],
                            audio_result["times_seconds"],
                            save_dir / "audio_attention.png",
                        )
                    ),
                    "temporal_timeline": str(
                        save_temporal_timeline_plot(
                            per_frame_probs.tolist(),
                            temporal_result.get("times_seconds"),
                            save_dir / "temporal_timeline.png",
                        )
                    ),
                }

        frame_times = temporal_result.get("times_seconds")

        # Multi-dimensional forensic diagnostics (anatomical saliency, texture, temporal stability)
        forensic_data = self.forensic_analyzer.generate_report_data(
            prediction="FAKE" if is_fake else "REAL",
            fake_probability=fake_probability,
            frame_probabilities=per_frame_probs.tolist(),
            frame_times=(
                frame_times.tolist() if frame_times is not None else None
            ),
            saliency_tensor=visual_saliency if visual_saliency_available else None,
            face_crops=face_crops,
            threshold=self.fake_threshold,
        )

        report = build_report(
            video_path=video_path,
            fake_probability=fake_probability,
            fake_probability_per_frame=per_frame_probs.tolist(),
            frame_times_seconds=(
                frame_times.tolist() if frame_times is not None else None
            ),
            visual_saliency_available=visual_saliency_available,
            audio_importance_available=audio_importance_available,
            threshold=self.fake_threshold,
            forensic_data=forensic_data,
        )
        report.visualization_paths = visualization_paths

        return report



    def close(self):
        self.face_detector.close()
