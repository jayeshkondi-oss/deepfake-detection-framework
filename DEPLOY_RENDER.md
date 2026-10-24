# Deploying Deepfake Detection Framework to Render

This step-by-step guide walks you through deploying your **DeepForensics AI** web application to **[Render](https://render.com)**.

---

## ⚠️ Important Note on Memory (RAM)
* This project runs a multimodal neural network combining **Vision Transformer (ViT-Base)** and **Meta Wav2Vec 2.0**.
* **Memory Requirement**: PyTorch + foundation models require ~**1.2 GB to 1.8 GB** of RAM during model loading and inference.
* **Instance Type**: 
  * Select the **Starter** instance plan (**2 GB RAM**, ~$7/month) on Render.
  * *Note: Render's Free tier is capped at 512 MB RAM, which will cause an `Out of Memory (OOM)` crash when loading large PyTorch models.*

---

## Step 1: Host Your 982 MB Model File (`best_model.pt`)

GitHub has a strict 100 MB file limit per commit, so `models/checkpoints/best_model.pt` (982 MB) cannot be directly pushed via standard Git.

Choose **one** of the following free options to get a direct download link:

### Option A: GitHub Releases (Recommended & Easiest)
1. Go to your GitHub repository: [github.com/jayeshkondi-oss/deepfake-detection-framework](https://github.com/jayeshkondi-oss/deepfake-detection-framework).
2. On the right sidebar, click **Releases** &rarr; **Create a new release** (or **Draft a new release**).
3. Set Tag: `v1.0.0` and Title: `Model Weights v1.0`.
4. In the **Attach binaries** box, drag and drop `models/checkpoints/best_model.pt` (982 MB) from your computer.
5. Click **Publish release**.
6. Right-click the uploaded `best_model.pt` asset and click **Copy link address**. (This is your `MODEL_URL`).

### Option B: Hugging Face Model Hub (Free & Unlimited)
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Model** (e.g. `your-username/deepforensics-checkpoint`).
3. Upload `best_model.pt` using the web UI or Git LFS.
4. Copy the direct download URL (`https://huggingface.co/.../resolve/main/best_model.pt`).

---

## Step 2: Deploy to Render

1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click the **New +** button in the top right &rarr; Select **Web Service**.
3. Choose **Build and deploy from a Git repository** &rarr; Click **Next**.
4. Connect your GitHub account and select **`jayeshkondi-oss/deepfake-detection-framework`**.
5. Configure the deployment settings:
   * **Name**: `deepforensics-ai` (or any name you choose)
   * **Region**: Choose closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   * **Branch**: `main`
   * **Runtime**: Select **Docker**
   * **Instance Type**: Select **Starter** (2 GB RAM, 1 CPU)
6. Scroll down to **Advanced** &rarr; **Environment Variables**:
   * Add Variable:
     * **Key**: `MODEL_URL`
     * **Value**: `<Your_Direct_Download_Link_from_Step_1>`
   * Add Variable:
     * **Key**: `PYTHONUNBUFFERED`
     * **Value**: `1`
7. Set **Health Check Path**:
   * Enter: `/api/health`
8. Click **Create Web Service**.

---

## What Happens During Build & Boot

1. **Docker Build**:
   - Render pulls your repository and builds the optimized `Dockerfile`.
   - It installs Linux dependencies (`ffmpeg`, `libgl1`, `libglib2.0-0`), sets up CPU-only PyTorch, and installs all packages.
2. **Container Startup**:
   - `app.py` runs and binds dynamically to Render's port (`PORT=10000`).
   - If `models/checkpoints/best_model.pt` is not present, `app.py` automatically downloads it from your `MODEL_URL` into `models/checkpoints/best_model.pt`.
   - The Vision Transformer and Wav2Vec2 backbones are initialized.
   - Render verifies `/api/health` and marks the service **Live**.
3. **Live Public URL**:
   - Your application will be live at:
     `https://deepforensics-ai.onrender.com` (or your chosen service name).

---

## Instant Local Public Demo (Alternative for Faculty Defense)

If you need a 100% free, immediate public URL to present to professors without paying for a cloud instance:
1. Double-click **`start_public_deployment.bat`** in this project folder.
2. It launches your local server on your GPU and outputs a secure public HTTPS URL (via localtunnel) that anyone can open on their laptop or smartphone.
