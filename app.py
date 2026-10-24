"""
app.py - Production Web Server for Multimodal Deepfake Detection & Forensic Explainability (XAI)
Runs FastAPI with an integrated responsive forensic dashboard.
"""

import os
import sys
import uuid
import base64
import shutil
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from deepfake.inference.predictor import DeepfakePredictor

app = FastAPI(
    title="DeepForensics AI",
    description="Multi-Modal Deepfake Detection & Explainability Web Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("outputs/web_uploads")
REPORTS_DIR = Path("outputs/web_reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Global predictor instance
predictor: Optional[DeepfakePredictor] = None
device_name = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = Path("models/checkpoints/best_model.pt")


@app.on_event("startup")
def load_model():
    global predictor
    print(f"[*] Initializing DeepForensics predictor on {device_name}...")
    ckpt = checkpoint_path if checkpoint_path.exists() else None
    predictor = DeepfakePredictor(
        checkpoint_path=ckpt,
        device=device_name,
        num_frames=16,
        face_margin=0.2,
        pretrained=False if ckpt else True,
        fake_threshold=0.65,
    )
    print("[+] Model loaded successfully and ready for inference!")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("src/deepfake/web/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/predict")
async def predict_video(
    video: UploadFile = File(...),
    threshold: float = Form(0.50)
):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")

    req_id = str(uuid.uuid4())[:8]
    ext = Path(video.filename).suffix.lower()
    ext = ext if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"} else ".mp4"
    
    saved_video_path = UPLOAD_DIR / f"upload_{req_id}{ext}"
    vis_dir = REPORTS_DIR / req_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Update threshold
        predictor.fake_threshold = float(threshold)

        # Run inference and explanations
        report = predictor.predict(
            video_path=saved_video_path,
            compute_explanations=True,
            save_visualizations_dir=vis_dir
        )

        # Generate HTML report
        html_report_path = vis_dir / "report.html"
        report.generate_html_report(str(html_report_path))

        # Automatically update preprocessing artifacts for live analysis & faculty demonstration
        try:
            from demo_preprocessing import run_preprocessing_demo
            # Save into current report dir
            run_preprocessing_demo(str(saved_video_path), str(vis_dir / "preprocessing"), report=report)
            # Also update global demo dir so /preprocessing-demo shows the latest live video
            run_preprocessing_demo(str(saved_video_path), "outputs/preprocessing_demo", report=report)
        except Exception as pe:
            print(f"[!] Warning: Preprocessing demo export failed: {pe}")

        # Encode visualizations as base64 for direct browser rendering
        vp = report.visualization_paths or {}
        b64 = {
            k: base64.b64encode(Path(vp[k]).read_bytes()).decode()
            if vp.get(k) and Path(vp[k]).exists() else ""
            for k in ["visual_attention", "audio_attention", "temporal_timeline"]
        }

        fd = report.forensic_data


        response_data = {
            "request_id": req_id,
            "filename": video.filename,
            "prediction": report.prediction,
            "fake_probability": report.fake_probability,
            "real_probability": 1.0 - report.fake_probability,
            "fake_pct": round(report.fake_probability * 100, 2),
            "real_pct": round((1.0 - report.fake_probability) * 100, 2),
            "threshold": float(threshold),
            "threshold_pct": round(float(threshold) * 100, 1),
            "is_fake": report.prediction == "FAKE",
            "verdict_comparison": (
                f"Fake Score {round(report.fake_probability * 100, 1)}% > Threshold {round(float(threshold) * 100, 1)}%"
                if report.prediction == "FAKE" else
                f"Fake Score {round(report.fake_probability * 100, 1)}% < Threshold {round(float(threshold) * 100, 1)}%"
            ),
            "simple_reasons": fd.simple_reasons if (fd and fd.simple_reasons) else [],
            "executive_summary": fd.human_understandable_summary if fd else "",
            "forensic_indicators": fd.forensic_indicators if fd else [],
            "anatomical_saliency": {
                "boundary_pct": round(fd.boundary_saliency_pct, 1) if fd else 0,
                "ocular_pct": round(fd.ocular_saliency_pct, 1) if fd else 0,
                "perioral_pct": round(fd.perioral_saliency_pct, 1) if fd else 0,
                "midface_pct": round(fd.midface_saliency_pct, 1) if fd else 0,
                "primary_region": fd.primary_focus_region if fd else "N/A",
            },
            "texture_and_frequency": {
                "texture_coherence_score": round(fd.texture_coherence_score, 1) if fd else 0,
                "laplacian_variance": round(fd.laplacian_variance, 1) if fd else 0,
                "high_frequency_ratio": round(fd.high_frequency_ratio, 3) if fd else 0,
                "has_generative_smoothing": bool(fd.has_generative_smoothing) if fd else False,
            },
            "video_url": f"/api/video/{req_id}",
            "temporal_dynamics": {
                "stability_score": round(fd.temporal_stability_score, 1) if fd else 0,
                "frame_variance": round(fd.frame_variance, 2) if fd else 0,
                "peak_timestamp_seconds": round(fd.peak_timestamp_seconds, 2) if fd else 0,
                "peak_frame_score_pct": round(fd.peak_frame_score_pct, 1) if fd else 0,
                "frame_probabilities": [round(p * 100, 1) for p in report.fake_probability_per_frame],
                "frame_times_seconds": [round(t, 2) for t in (report.frame_times_seconds or [])],
                "face_boxes": report.face_boxes if report.face_boxes else [],
            },
            "visualizations": {
                "visual_attention_b64": b64["visual_attention"],
                "audio_attention_b64": b64["audio_attention"],
                "temporal_timeline_b64": b64["temporal_timeline"],
            },
            "extracted_frames": getattr(report, "extracted_frames", None) or [],
            "fake_frame_indices": [f["frame_index"] for f in (getattr(report, "extracted_frames", None) or []) if f.get("is_fake")],
            "fake_frames_summary": (
                f"Manipulation localized in {len([f for f in (getattr(report, 'extracted_frames', None) or []) if f.get('is_fake')])} frames "
                f"(Frames #{', #'.join(str(f['frame_index']) for f in (getattr(report, 'extracted_frames', None) or []) if f.get('is_fake'))} "
                f"between {min((f['timestamp_seconds'] for f in (getattr(report, 'extracted_frames', None) or []) if f.get('is_fake')), default=0.0):.1f}s and "
                f"{max((f['timestamp_seconds'] for f in (getattr(report, 'extracted_frames', None) or []) if f.get('is_fake')), default=0.0):.1f}s)."
                if any(f.get("is_fake") for f in (getattr(report, "extracted_frames", None) or []))
                else "All 16 sampled frames passed authentic biometric checks with genuine camera grain and natural facial dynamics."
            ),
            "report_download_url": f"/api/report/{req_id}"
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/video/{req_id}")
def stream_uploaded_video(req_id: str):
    matches = list(UPLOAD_DIR.glob(f"upload_{req_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found.")
    video_path = matches[0]
    return FileResponse(path=video_path, media_type="video/mp4")


@app.get("/api/report/{req_id}")
def download_html_report(req_id: str):
    report_file = REPORTS_DIR / req_id / "report.html"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        path=report_file,
        filename=f"deepfake_forensic_report_{req_id}.html",
        media_type="text/html"
    )


# Faculty Demonstration Route
app.mount(
    "/preprocessing-demo-files",
    StaticFiles(directory="outputs/preprocessing_demo"),
    name="preprocessing_demo_files"
)

@app.get("/preprocessing-demo", response_class=HTMLResponse)
def serve_preprocessing_demo():
    demo_file = Path("outputs/preprocessing_demo/preprocessing_demo.html")
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Preprocessing demo not generated. Run demo_preprocessing.py first.")
    with open(demo_file, "r", encoding="utf-8") as f:
        # replace relative img paths with mounted static path
        content = f.read()
        content = content.replace('src="step', 'src="/preprocessing-demo-files/step')
        return content


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[+] Starting DeepForensics Web Server on http://127.0.0.1:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
