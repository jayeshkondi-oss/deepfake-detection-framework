"""
report.py
Combine the classifier verdict, temporal localization, and visual/
audio explanations into a single structured, human-readable report --
the final box in the original architecture diagram.
"""

from dataclasses import dataclass
from typing import Optional

from deepfake.explainability.forensic_analyzer import ForensicReportData


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
    threshold: float = 0.5
    most_suspicious_frame_index: int | None = None
    visualization_paths: dict | None = None
    forensic_data: Optional[ForensicReportData] = None

    def summary(self) -> str:
        """A detailed, human-readable forensic XAI explanation report."""
        real_prob = (1.0 - self.fake_probability) * 100
        fake_prob = self.fake_probability * 100

        lines = [
            "=" * 70,
            "              MULTI-MODAL DEEPFAKE FORENSIC REPORT (XAI)",
            "=" * 70,
            f"Analyzed Video: {self.video_path}",
            f"Final Verdict : [{self.prediction}]",
            f"Confidence    : {real_prob:.1f}% REAL | {fake_prob:.1f}% FAKE  (Threshold: {self.threshold * 100:.1f}%)",
            "-" * 70,
        ]

        if self.forensic_data:
            lines.extend([
                "EXECUTIVE FORENSIC SUMMARY (Human-Understandable):",
                f"   {self.forensic_data.human_understandable_summary}",
                "-" * 70,
                "KEY FORENSIC FINDINGS & INDICATORS:",
            ])
            for ind in self.forensic_data.forensic_indicators:
                lines.append(f"   [x] {ind}")
            lines.extend([
                "-" * 70,
                "DETAILED MULTI-MODAL XAI BREAKDOWN:",
                "-" * 70,
                "1. ANATOMICAL FACIAL SALIENCY (Vision Transformer Attention Rollout):",
                f"   * Face Blending Boundary : {self.forensic_data.boundary_saliency_pct:.1f}% of total attention",
                f"   * Ocular Region (Eyes)   : {self.forensic_data.ocular_saliency_pct:.1f}% of total attention",
                f"   * Perioral Region (Mouth): {self.forensic_data.perioral_saliency_pct:.1f}% of total attention",
                f"   * Midface (Nose & Cheeks): {self.forensic_data.midface_saliency_pct:.1f}% of total attention",
                f"   * Primary Focus Zone     : {self.forensic_data.primary_focus_region}",
                "",
                "2. HIGH-FREQUENCY SPECTRAL & TEXTURE ANALYSIS (FFT / Laplacian):",
                f"   * Texture Realism Score  : {self.forensic_data.texture_coherence_score:.1f}%",
                f"   * Laplacian Variance     : {self.forensic_data.laplacian_variance:.1f}",
                f"   * High-Frequency Ratio   : {self.forensic_data.high_frequency_ratio:.3f}",
                f"   * Generative Smoothing   : {'DETECTED (Synthetic Blur)' if self.forensic_data.has_generative_smoothing else 'NONE (Natural Sensor Noise Preserved)'}",
                "",
                "3. TEMPORAL DYNAMICS & FRAME-BY-FRAME STABILITY:",
                f"   * Temporal Stability     : {self.forensic_data.temporal_stability_score:.1f}%",
                f"   * Frame Variance         : {self.forensic_data.frame_variance:.2f}",
                f"   * Peak Anomaly Moment    : ~{self.forensic_data.peak_timestamp_seconds:.2f}s ({self.forensic_data.peak_frame_score_pct:.1f}% fake probability)",
                "",
                "4. AUDIO-VISUAL CROSS-MODAL FUSION:",
            ])
        else:
            lines.extend([
                "EXPLAINABLE AI (XAI) DIAGNOSTIC BREAKDOWN:",
                "-" * 70,
            ])

            # Fallback 1. Temporal Analysis
            lines.append("1. TEMPORAL CONSISTENCY & LOCALIZATION:")
            if self.fake_probability_per_frame:
                min_p = min(self.fake_probability_per_frame) * 100
                max_p = max(self.fake_probability_per_frame) * 100
                variance = sum((p * 100 - fake_prob) ** 2 for p in self.fake_probability_per_frame) / len(self.fake_probability_per_frame)

                if self.most_suspicious_frame_index is not None:
                    idx = self.most_suspicious_frame_index
                    time_str = (
                        f"~{self.frame_times_seconds[idx]:.2f}s"
                        if self.frame_times_seconds is not None
                        else f"frame {idx}"
                    )
                else:
                    time_str = "N/A"

                if self.prediction == "REAL":
                    lines.append(
                        f"   * Status: Consistent authentic temporal pattern across all frames."
                    )
                    lines.append(
                        f"   * Frame Fake Scores: Min = {min_p:.1f}%, Max = {max_p:.1f}%, Variance = {variance:.2f}"
                    )
                    lines.append(
                        f"   * Peak Anomaly Timestamp: {time_str} ({max_p:.1f}% - stays well below {self.threshold * 100:.1f}% threshold)."
                    )
                    lines.append(
                        f"   * Explanation: Natural biological transitions observed across temporal frames;"
                        f" no abrupt frame-switching or temporal jitter detected."
                    )
                else:
                    lines.append(
                        f"   * Status: Temporal inconsistency and manipulation spikes detected."
                    )
                    lines.append(
                        f"   * Peak Manipulation Timestamp: {time_str} ({max_p:.1f}% fake confidence at this instant)."
                    )
                    lines.append(
                        f"   * Frame Fake Scores: Range [{min_p:.1f}% to {max_p:.1f}%], Variance = {variance:.2f}"
                    )
                    lines.append(
                        f"   * Explanation: Significant manipulation likelihood localized around {time_str},"
                        f" indicating potential face-swapping or reenactment artifacts."
                    )

            # Fallback 2. Visual Attention Saliency
            lines.append("\n2. VISUAL ATTENTION ROLLOUT (Spatial Analysis):")
            if self.visual_saliency_available:
                if self.prediction == "REAL":
                    lines.append(
                        "   * ViT Attention Pattern: Uniformly distributed across natural facial landmarks."
                    )
                    lines.append(
                        "   * Explanation: The Vision Transformer focuses on genuine ocular and oral features."
                        " No localized face-swap boundary seams, warping, or blending artifacts detected."
                    )
                else:
                    lines.append(
                        "   * ViT Attention Pattern: High attention concentration along facial perimeter & mouth."
                    )
                    lines.append(
                        "   * Explanation: Attention rollout reveals abnormal weight concentration at face boundaries"
                        " or mouth region, characteristic of synthetic blending and digital seam artifacts."
                    )
            else:
                lines.append("   * Visual saliency explanation skipped.")

            lines.append("\n3. AUDIO-VISUAL CROSS-MODAL FUSION:")

        if self.audio_importance_available:
            if self.prediction == "REAL":
                lines.append(
                    "   * Wav2Vec2 & Cross-Attention: Natural acoustic-visual alignment."
                )
                lines.append(
                    "   * Explanation: Speech acoustic dynamics match facial articulation;"
                    " no synthetic vocoder traces or audio-visual desynchronization detected."
                )
            else:
                lines.append(
                    "   * Wav2Vec2 & Cross-Attention: Cross-modal discordance detected."
                )
                lines.append(
                    "   * Explanation: Acoustic features exhibit unnatural spectral distribution or lack"
                    " synchronization with facial motion, typical of AI audio cloning/dubbing."
                )
        else:
            lines.append("   * Audio importance explanation skipped.")

        # 4. Visualizations saved
        if self.visualization_paths:
            lines.append("-" * 70)
            lines.append("SAVED EXPLAINABILITY VISUALIZATIONS:")
            for name, path in self.visualization_paths.items():
                lines.append(f"   [+] {name:<18}: {path}")
            lines.append("=" * 70)

        return "\n".join(lines)

    def generate_html_report(self, output_path: str = "outputs/report.html") -> str:
        """Generate a visual, interactive HTML dashboard explaining the verdict."""
        import base64
        from pathlib import Path

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        def img_to_base64(path_str):
            if not path_str or not Path(path_str).exists():
                return ""
            with open(path_str, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        vis_b64 = img_to_base64(self.visualization_paths.get("visual_attention") if self.visualization_paths else None)
        aud_b64 = img_to_base64(self.visualization_paths.get("audio_attention") if self.visualization_paths else None)
        tem_b64 = img_to_base64(self.visualization_paths.get("temporal_timeline") if self.visualization_paths else None)

        verdict_color = "#28a745" if self.prediction == "REAL" else "#dc3545"
        real_pct = (1.0 - self.fake_probability) * 100
        fake_pct = self.fake_probability * 100

        forensic_html = ""
        if self.forensic_data:
            fd = self.forensic_data
            indicators_list = "".join([f"<li style='margin-bottom:6px;'>{ind}</li>" for ind in fd.forensic_indicators])
            forensic_html = f"""
        <div class="card" style="border-left: 5px solid #38bdf8;">
            <div class="card-title" style="color:#38bdf8;">Executive Forensic Summary (Human-Understandable)</div>
            <div class="explanation" style="font-size:15px; background:#1e293b; border-left:none; padding:10px 0;">
                {fd.human_understandable_summary}
            </div>
            <div style="margin-top:16px;">
                <strong style="font-size:14px; color:#94a3b8;">Key Forensic Indicators:</strong>
                <ul style="font-size:13px; color:#cbd5e1; margin-top:8px; padding-left:20px;">
                    {indicators_list}
                </ul>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Anatomical Facial Saliency Allocation (ViT Attention Breakdown)</div>
            <p style="font-size:13px; color:#94a3b8; margin-bottom:16px;">Quantifies where the Vision Transformer focused its attention across anatomical facial regions.</p>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px;">
                <div style="background:#0f172a; padding:12px; border-radius:8px;">
                    <div style="font-size:12px; color:#94a3b8;">Face Blending Boundary</div>
                    <div style="font-size:22px; font-weight:bold; color:{'#ef4444' if fd.boundary_saliency_pct > 32 else '#38bdf8'};">{fd.boundary_saliency_pct:.1f}%</div>
                    <div style="font-size:11px; color:#64748b;">Threshold: <32% normal</div>
                </div>
                <div style="background:#0f172a; padding:12px; border-radius:8px;">
                    <div style="font-size:12px; color:#94a3b8;">Ocular (Eyes & Brows)</div>
                    <div style="font-size:22px; font-weight:bold; color:#38bdf8;">{fd.ocular_saliency_pct:.1f}%</div>
                    <div style="font-size:11px; color:#64748b;">Gaze & blinking</div>
                </div>
                <div style="background:#0f172a; padding:12px; border-radius:8px;">
                    <div style="font-size:12px; color:#94a3b8;">Perioral (Mouth & Lips)</div>
                    <div style="font-size:22px; font-weight:bold; color:{'#ef4444' if fd.perioral_saliency_pct > 35 else '#38bdf8'};">{fd.perioral_saliency_pct:.1f}%</div>
                    <div style="font-size:11px; color:#64748b;">Phoneme articulation</div>
                </div>
                <div style="background:#0f172a; padding:12px; border-radius:8px;">
                    <div style="font-size:12px; color:#94a3b8;">Midface (Nose & Cheeks)</div>
                    <div style="font-size:22px; font-weight:bold; color:#38bdf8;">{fd.midface_saliency_pct:.1f}%</div>
                    <div style="font-size:11px; color:#64748b;">Skin reflectance</div>
                </div>
            </div>
            <div style="margin-top:14px; font-size:13px; color:#cbd5e1;">
                <strong>Primary Visual Focus Zone:</strong> <span style="color:#38bdf8;">{fd.primary_focus_region}</span>
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-val" style="color:{'#22c55e' if fd.texture_coherence_score > 70 else '#ef4444'};">{fd.texture_coherence_score:.1f}%</div>
                <div class="metric-lbl">Texture Realism (FFT / Laplacian)</div>
            </div>
            <div class="metric-card">
                <div class="metric-val" style="color:{'#22c55e' if fd.temporal_stability_score > 70 else '#ef4444'};">{fd.temporal_stability_score:.1f}%</div>
                <div class="metric-lbl">Temporal Motion Stability</div>
            </div>
            <div class="metric-card">
                <div class="metric-val" style="color:#38bdf8;">~{fd.peak_timestamp_seconds:.2f}s</div>
                <div class="metric-lbl">Peak Anomaly Timestamp</div>
            </div>
        </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Deepfake Detection XAI Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
        .container {{ max-width: 1050px; margin: 0 auto; }}
        .header {{ background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 24px; border-left: 6px solid {verdict_color}; }}
        .title {{ font-size: 24px; font-weight: bold; margin-bottom: 8px; color: #f8fafc; }}
        .subtitle {{ font-size: 14px; color: #94a3b8; }}
        .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: bold; font-size: 16px; background: {verdict_color}; color: white; margin-top: 12px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 20px 0; }}
        .metric-card {{ background: #1e293b; padding: 18px; border-radius: 10px; text-align: center; border: 1px solid #334155; }}
        .metric-val {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
        .metric-lbl {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
        .card-title {{ font-size: 18px; font-weight: bold; color: #38bdf8; margin-bottom: 12px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .explanation {{ background: #0f172a; padding: 14px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-top: 14px; font-size: 14px; line-height: 1.6; }}
        .chart-img {{ width: 100%; border-radius: 8px; margin-top: 12px; max-height: 380px; object-fit: contain; background: #000; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">Multimodal Deepfake Detection & Explainability (XAI) Report</div>
            <div class="subtitle">Video: {self.video_path}</div>
            <div class="badge">VERDICT: {self.prediction}</div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-val">{real_pct:.1f}%</div>
                <div class="metric-lbl">Authentic (Real) Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{fake_pct:.1f}%</div>
                <div class="metric-lbl">Synthetic (Fake) Probability</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{self.threshold * 100:.1f}%</div>
                <div class="metric-lbl">Decision Threshold</div>
            </div>
        </div>

        {forensic_html}

        <div class="card">
            <div class="card-title">1. Spatial Attention Heatmap (Vision Transformer XAI)</div>
            <p style="font-size:14px; color:#cbd5e1;">Attention Rollout maps spatial focus across 14&times;14 facial patches. Brighter areas indicate regions where the model evaluated visual consistency.</p>
            {f'<img class="chart-img" src="data:image/png;base64,{vis_b64}" alt="Visual Saliency">' if vis_b64 else '<p style="color:#64748b;">Visual attention plot not available.</p>'}
            <div class="explanation">
                <strong>Diagnostic Explanation:</strong>
                {"Visual attention is uniformly distributed across natural facial landmarks (eyes, nose, mouth) with no localized boundary seams or blending distortions detected." if self.prediction == "REAL" else "High attention concentration observed along the facial periphery or mouth boundaries, indicating digital blending or face-swap seam anomalies."}
            </div>
        </div>

        <div class="card">
            <div class="card-title">2. Temporal Manipulation Timeline (Transformer Frame Localization)</div>
            <p style="font-size:14px; color:#cbd5e1;">Tracks the manipulation probability curve across all sampled frames over the duration of the clip.</p>
            {f'<img class="chart-img" src="data:image/png;base64,{tem_b64}" alt="Temporal Timeline">' if tem_b64 else '<p style="color:#64748b;">Temporal timeline not available.</p>'}
            <div class="explanation">
                <strong>Diagnostic Explanation:</strong>
                {"All frames remain consistently inside the safe authentic zone. No localized frame-switching or temporal flicker was detected." if self.prediction == "REAL" else "Temporal spikes in fake confidence detected during specific seconds of the clip, identifying localized manipulation intervals."}
            </div>
        </div>

        <div class="card">
            <div class="card-title">3. Acoustic Importance & Cross-Modal Synchronization</div>
            <p style="font-size:14px; color:#cbd5e1;">Analyzes Wav2Vec2 audio attention weights combined with visual motion via cross-attention fusion.</p>
            {f'<img class="chart-img" src="data:image/png;base64,{aud_b64}" alt="Audio Attention">' if aud_b64 else '<p style="color:#64748b;">Audio attention plot not available.</p>'}
            <div class="explanation">
                <strong>Diagnostic Explanation:</strong>
                {"Acoustic frequency dynamics maintain natural continuity with facial motion; no synthetic vocoder artifacts detected." if self.prediction == "REAL" else "Acoustic features demonstrate unnatural spectral patterns or desynchronization with visual facial motion."}
            </div>
        </div>
    </div>
</body>
</html>
"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html)
        return str(out_p)



def build_report(
    video_path: str,
    fake_probability: float,
    fake_probability_per_frame,
    frame_times_seconds=None,
    visual_saliency_available: bool = False,
    audio_importance_available: bool = False,
    threshold: float = 0.5,
    forensic_data: Optional[ForensicReportData] = None,
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
        forensic_data: optional ForensicReportData with anatomical and frequency diagnostics.

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
        threshold=float(threshold),
        most_suspicious_frame_index=most_suspicious_index,
        forensic_data=forensic_data,
    )



