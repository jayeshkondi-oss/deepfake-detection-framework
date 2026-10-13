"""
demo_preprocessing.py
Interactive Preprocessing Visualizer for Academic/Faculty Demonstrations.

Demonstrates:
1. Video Uniform Temporal Frame Sampling (Decord)
2. Face Detection & 20% Margin Expansion (MediaPipe)
3. 224x224 Tensor Resizing & Normalization (OpenCV + PyTorch)
4. Audio Track Demuxing & 16kHz Waveform Windowing (FFmpeg + PyTorch)
5. Generates an interactive visual HTML dashboard with side-by-side comparisons.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepfake.data.video_loader import VideoLoader
from deepfake.data.face_detector import FaceDetector
from deepfake.data.preprocessing import FramePreprocessor
from deepfake.data.audio_window import AudioWindowExtractor


def run_preprocessing_demo(video_path: str, output_dir: str = "outputs/preprocessing_demo"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_path = str(video_path)

    print("=" * 65)
    print("      DEEPFAKE DETECTION FRAMEWORK: PREPROCESSING DEMO       ")
    print("=" * 65)
    print(f"[*] Input Video: {video_path}")

    # -------------------------------------------------------------
    # 1. Video Loading & Uniform Sampling
    # -------------------------------------------------------------
    print("\n[Step 1] Loading video and uniformly sampling 16 frames via Decord...")
    with VideoLoader(video_path) as loader:
        fps = loader.fps
        total_frames = loader.frame_count
        duration = total_frames / fps if fps > 0 else 0.0
        raw_frames = loader.sample_frames(16)

    print(f"    -> Original Resolution: {raw_frames[0].shape[1]}x{raw_frames[0].shape[0]}")
    print(f"    -> Native Video FPS:    {fps:.2f}")
    print(f"    -> Total Video Frames:  {total_frames}")
    print(f"    -> Duration:            {duration:.2f} seconds")
    print(f"    -> Sampled Frames:      16 uniformly spaced across 0% to 100%")

    # -------------------------------------------------------------
    # 2. Face Detection with 20% Margin Expansion
    # -------------------------------------------------------------
    print("\n[Step 2] Detecting facial landmarks & applying 20% margin (MediaPipe)...")
    face_detector = FaceDetector(min_confidence=0.25)
    detected_boxes = []
    face_crops = []
    annotated_frames = []

    for idx, frame in enumerate(raw_frames):
        rgb_frame = frame[:, :, ::-1]
        detections = face_detector.detect(rgb_frame)
        annotated = frame.copy()

        if detections:
            best_det = max(detections, key=lambda d: d.confidence)
            box = best_det.bbox  # (x1, y1, x2, y2)
            detected_boxes.append(box)

            # Draw original tight bounding box (Blue)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 100, 0), 2)
            cv2.putText(annotated, f"Tight BBox ({best_det.confidence*100:.0f}%)", 
                        (box[0], max(20, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            # Crop with 20% margin
            crop = face_detector.crop_face(frame, best_det, margin=0.2)
            face_crops.append(crop)

            # Calculate and draw 20% margin box (Green)
            h, w = frame.shape[:2]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            x1_m = max(0, int(box[0] - 0.2 * bw))
            y1_m = max(0, int(box[1] - 0.2 * bh))
            x2_m = min(w, int(box[2] + 0.2 * bw))
            y2_m = min(h, int(box[3] + 0.2 * bh))
            cv2.rectangle(annotated, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 0), 3)
            cv2.putText(annotated, "Crop (+20% Margin: Hairline & Jaw)", 
                        (x1_m, min(h - 10, y2_m + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            face_crops.append(frame)
            cv2.putText(annotated, "No Face Detected (Fallback: Full Frame)", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        annotated_frames.append(annotated)

    # Save a comparison image of Frame 0
    sample_annotated_path = out_path / "step2_face_detection_comparison.jpg"
    cv2.imwrite(str(sample_annotated_path), annotated_frames[0])
    print(f"    -> Saved Bounding Box Comparison to: {sample_annotated_path}")

    # -------------------------------------------------------------
    # 3. Resizing & Normalization into PyTorch Tensor [16, 3, 224, 224]
    # -------------------------------------------------------------
    print("\n[Step 3] Bilinear resizing to 224x224 and ImageNet normalization...")
    preprocessor = FramePreprocessor(image_size=224)
    frames_tensor = preprocessor.process_frames(face_crops)

    print(f"    -> Output Tensor Shape: {list(frames_tensor.shape)}")
    print(f"       [Batch Frames=16, RGB Channels=3, Height=224, Width=224]")
    print(f"    -> Tensor Data Type:    {frames_tensor.dtype}")
    print(f"    -> Tensor Value Range:  Min={frames_tensor.min():.2f}, Max={frames_tensor.max():.2f}")

    # Save 16-crop montage grid
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle("Step 3: 16 Uniform Temporal Face Crops (Resized to 224x224 px for ViT-Base)", fontsize=14, fontweight='bold')
    for i in range(16):
        ax = axes[i // 8, i % 8]
        crop_rgb = cv2.cvtColor(face_crops[i], cv2.COLOR_BGR2RGB)
        crop_res = cv2.resize(crop_rgb, (224, 224))
        ax.imshow(crop_res)
        ax.set_title(f"T={i+1}/16", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    montage_path = out_path / "step3_16_crops_montage.png"
    plt.savefig(str(montage_path), dpi=150)
    plt.close()
    print(f"    -> Saved 16-Frame Face Grid to: {montage_path}")

    # -------------------------------------------------------------
    # 4. Audio Extraction & Standardization [64000 samples @ 16kHz]
    # -------------------------------------------------------------
    print("\n[Step 4] Demuxing audio track & windowing to 4.0s @ 16kHz...")
    audio_extractor = AudioWindowExtractor(sample_rate=16000, duration_seconds=4.0)
    audio_tensor = audio_extractor.extract(video_path)

    print(f"    -> Output Audio Shape:  {list(audio_tensor.shape)}")
    print(f"       [64,000 samples = 4.0 seconds @ 16,000 Hz Mono]")
    print(f"    -> Audio Max Amplitude: {audio_tensor.abs().max():.4f}")

    # Plot Audio Waveform & Spectrogram
    audio_np = audio_tensor.numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle("Step 4: Standardized Audio Track (16,000 Hz, 4.0s for Wav2Vec 2.0)", fontsize=14, fontweight='bold')

    time_axis = np.linspace(0, 4.0, len(audio_np))
    ax1.plot(time_axis, audio_np, color="#2563eb", lw=1)
    ax1.set_title("Time-Domain Waveform (64,000 Float Samples)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim([-1.05, 1.05])
    ax1.grid(True, alpha=0.3)

    # Spectrogram
    ax2.specgram(audio_np, Fs=16000, NFFT=512, noverlap=256, cmap="viridis")
    ax2.set_title("Acoustic Frequency Spectrogram (0 to 8,000 Hz)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    audio_plot_path = out_path / "step4_audio_waveform.png"
    plt.savefig(str(audio_plot_path), dpi=150)
    plt.close()
    print(f"    -> Saved Audio Waveform & Spectrogram to: {audio_plot_path}")

    # -------------------------------------------------------------
    # 5. Build Academic HTML Presentation Dashboard
    # -------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Deepfake Preprocessing Pipeline - Faculty Demo</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0b0f19; color: #f1f5f9; margin: 0; padding: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 14px; padding: 28px; border-left: 6px solid #38bdf8; box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin-bottom: 25px; }}
        h1 {{ margin: 0 0 10px 0; font-size: 26px; color: #38bdf8; }}
        .step-card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 25px; border: 1px solid #334155; }}
        h2 {{ font-size: 20px; color: #f8fafc; margin-top: 0; display: flex; align-items: center; gap: 10px; }}
        .badge {{ background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: bold; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .code-box {{ background: #0f172a; border-radius: 8px; padding: 15px; font-family: Consolas, monospace; font-size: 13px; color: #38bdf8; border: 1px solid #1e293b; }}
        img {{ max-width: 100%; border-radius: 8px; border: 1px solid #475569; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 14px; border-bottom: 1px solid #334155; text-align: left; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
        .pill {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; background: #22c55e20; color: #4ade80; border: 1px solid #22c55e50; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔬 Deepfake Detection Preprocessing Architecture</h1>
        <p style="color: #94a3b8; margin: 0;">Interactive demonstration of data calibration, facial tracking, and acoustic demuxing before neural tensor ingestion.</p>
        <p style="margin: 8px 0 0 0; font-size: 13px; color: #cbd5e1;"><b>Video Analyzed:</b> {Path(video_path).name} &nbsp;|&nbsp; <b>Resolution:</b> {raw_frames[0].shape[1]}x{raw_frames[0].shape[0]} &nbsp;|&nbsp; <b>Duration:</b> {duration:.2f}s ({total_frames} frames)</p>
    </div>

    <!-- Step 1 & 2 -->
    <div class="step-card">
        <h2><span class="badge">STEP 1 & 2</span> Video Decoding & Face Tracking with 20% Margin</h2>
        <div class="grid-2">
            <div>
                <img src="step2_face_detection_comparison.jpg" alt="Face Detection with Margin">
                <p style="font-size: 13px; color: #94a3b8; margin-top: 8px;">
                    <b style="color: #38bdf8;">Blue Box:</b> Tight landmark bbox (cuts off jaw & hairline).<br/>
                    <b style="color: #4ade80;">Green Box:</b> 20% expanded margin crop (captures cut-lines & blending seams).
                </p>
            </div>
            <div>
                <p><b>Why MediaPipe + 20% Margin?</b></p>
                <ul style="color: #cbd5e1; font-size: 14px; line-height: 1.6;">
                    <li><b>Eliminates Distractions:</b> Filters out backgrounds, clothing, and irrelevant motion so the model focuses 100% on the face.</li>
                    <li><b>Preserves Boundary Artifacts:</b> Deepfake face-swapping algorithms leave their primary feathering and blending errors at the forehead hairline and under the chin. Tight crops delete this evidence; the 20% margin keeps it intact.</li>
                    <li><b>Uniform Temporal Decord Sampling:</b> Extracts 16 frames spaced evenly across the full duration rather than just the first second.</li>
                </ul>
                <div class="code-box">
                    Frames Sampled: 16 uniformly spaced<br/>
                    Sampling Interval: ~{total_frames/16:.1f} frames apart<br/>
                    Detector: MediaPipe Face SSD MobileNet<br/>
                    Expansion Margin: 0.2 (20% on each side)
                </div>
            </div>
        </div>
    </div>

    <!-- Step 3 -->
    <div class="step-card">
        <h2><span class="badge">STEP 3</span> 16 Uniform Temporal Face Crops (Resized to 224x224 px)</h2>
        <img src="step3_16_crops_montage.png" alt="16 Face Crops Montage">
        <table style="margin-top: 20px;">
            <tr><th>Parameter</th><th>Value</th><th>Technical Rationale for Faculty</th></tr>
            <tr><td><b>NUM_FRAMES</b></td><td><span class="pill">16 Frames</span></td><td>Captures complete blink cycles (100–400ms) and head turns across full video span while staying within GPU VRAM limits.</td></tr>
            <tr><td><b>IMAGE_SIZE</b></td><td><span class="pill">224 x 224 px</span></td><td>Matches pretrained Vision Transformer (ViT-Base) patch grid: 224 / 16 = 14 patches &rarr; 196 spatial tokens.</td></tr>
            <tr><td><b>Tensor Shape</b></td><td><span class="pill">[16, 3, 224, 224]</span></td><td>Normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).</td></tr>
        </table>
    </div>

    <!-- Step 4 -->
    <div class="step-card">
        <h2><span class="badge">STEP 4</span> Audio Standardization (16,000 Hz, 4.0 Seconds)</h2>
        <div class="grid-2">
            <div>
                <img src="step4_audio_waveform.png" alt="Audio Waveform & Spectrogram">
            </div>
            <div>
                <p><b>Why 16kHz & 4.0-Second Window?</b></p>
                <ul style="color: #cbd5e1; font-size: 14px; line-height: 1.6;">
                    <li><b>Wav2Vec 2.0 Compatibility:</b> Modern speech foundation models are strictly pretrained on 16 kHz audio. Feeding 44.1k/48k directly causes severe frequency distortion ($3\times$ speedup).</li>
                    <li><b>Nyquist Theorem:</b> 16 kHz sampling captures frequencies up to 8,000 Hz ($16000 / 2$), which covers 100% of human voice formants (F1–F4) while cutting file size by 66%.</li>
                    <li><b>Token Alignment:</b> 4.0 seconds @ 16 kHz = 64,000 samples &rarr; 200 Wav2Vec tokens, cleanly balancing the 16 visual tokens in Cross-Attention.</li>
                </ul>
                <div class="code-box">
                    Audio Sample Rate: 16,000 Hz (Mono PCM)<br/>
                    Audio Duration: 4.0 seconds (Center-cropped or Zero-padded)<br/>
                    Audio Tensor Shape: torch.Size([64000])<br/>
                    Acoustic Bandwidth: 0 to 8,000 Hz
                </div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""

    html_path = out_path / "preprocessing_demo.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 65)
    print("           PREPROCESSING DEMO GENERATION COMPLETE!           ")
    print("=" * 65)
    print(f"[+] Output Directory: {out_path.resolve()}")
    print(f"[+] Interactive HTML: {html_path.resolve()}")
    print("=" * 65)
    return str(html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Preprocessing Demonstration on any video.")
    parser.add_argument("--video", type=str, default="outputs/web_uploads/upload_9dbd9ae9.mp4",
                        help="Path to video file to demonstrate preprocessing on.")
    parser.add_argument("--output_dir", type=str, default="outputs/preprocessing_demo",
                        help="Folder to save visual demonstration artifacts.")
    args = parser.parse_args()
    run_preprocessing_demo(args.video, args.output_dir)
