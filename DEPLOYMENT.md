# 🚀 DeepForensics AI - Deployment Guide

This guide explains how to run the web application locally and deploy it online to cloud platforms (Hugging Face Spaces, Render, AWS, Railway).

---

## 1. Local Deployment (Windows / Linux / Mac)

### Quick Start (Windows):
Simply double-click:
```cmd
run_app.bat
```
Or run in PowerShell:
```powershell
.venv\Scripts\python.exe app.py
```
Open your browser at:
👉 **`http://localhost:8000`**

### Features on the Website:
- **Drag & Drop Upload**: Upload any video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
- **Interactive Video Preview**: Preview and playback the uploaded clip before analyzing.
- **Adjustable Sensitivity Slider**: Configure decision threshold (Default: 0.65 for unseen videos).
- **Executive Forensic Diagnosis**: Plain-English, human-understandable explanation for judges/professors.
- **Key Forensic Indicators**: Clear anomaly bullet tags.
- **Anatomical Facial Saliency Breakdown**: ViT attention progress bars for Blending Boundary, Eyes, Mouth, and Midface.
- **Deep Explanatory Visualizations**:
  - Spatial Attention Heatmap (ViT Rollout over suspicious frame)
  - Temporal Manipulation Timeline (Frame-by-frame probability curve)
  - Acoustic Importance (Wav2Vec2 alignment)
- **Direct HTML Report Download**: One-click download of the complete standalone forensic report.
- **REST API & Swagger Docs**: Interactive API documentation at `http://localhost:8000/docs`.

---

## 2. Free Cloud Deployment (Hugging Face Spaces)

Hugging Face Spaces provides free hosting with optional GPU acceleration.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space SDK to **Docker** (Blank).
3. Connect your GitHub repository:
   `https://github.com/jayeshkondi-oss/deepfake-detection-framework`
4. The included `Dockerfile` will automatically build the environment, install PyTorch, MediaPipe, OpenCV, and launch `app.py`.
5. Your public web app will be live at:
   `https://huggingface.co/spaces/<your-username>/deepfake-detection-framework`

---

## 3. Render.com / Railway Deployment

1. Create a free account on [Render.com](https://render.com) or [Railway.app](https://railway.app).
2. Click **New Web Service** and link your GitHub repository.
3. Choose **Docker** as the Environment.
4. Set the Port to `8000`.
5. Click **Deploy**. Render will build and provide an instant public HTTPS URL (e.g. `https://deepforensics.onrender.com`).

---

## 4. Docker (Self-Hosted Linux / VPS / AWS EC2)

Build and run with Docker:
```bash
docker build -t deepforensics-app .
docker run -d -p 8000:8000 --name deepforensics deepforensics-app
```
Then visit `http://<your-server-ip>:8000`.
