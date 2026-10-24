# Deploying Deepfake Detection Framework to Railway (Cloud PaaS)

This guide walks you through deploying your **DeepForensics AI** framework to **Railway** so anyone can access your live deepfake detection web application from any browser.

---

## Why Railway?
* **Generous RAM**: Railway provides up to 8GB of RAM dynamically, preventing Out-Of-Memory (OOM) errors during Vision Transformer (ViT) and Wav2Vec2 audio inference.
* **Native Docker Support**: Automatically builds using the optimized production `Dockerfile`.
* **Free HTTPS Public URL**: Provides a permanent `https://your-app.up.railway.app` URL with automatic SSL.

---

## Method 1: Direct Deploy via Railway CLI (Recommended & Easiest)

Because your trained model (`models/checkpoints/best_model.pt`) is **982 MB**, pushing it to GitHub directly exceeds GitHub's 100MB file limit. 
Using the **Railway CLI** allows you to upload and deploy your local model weights and code straight to Railway without GitHub file size restrictions.

### Step 1: Install Railway CLI
Open PowerShell or Command Prompt on your computer and run:
```bash
npm install -g @railway/cli
```

### Step 2: Login to Railway
```bash
railway login
```
*(This opens your browser to authorize your Railway account for free.)*

### Step 3: Initialize and Deploy
Inside your project folder (`c:\Users\jayes\OneDrive\Desktop\deepfake-detection-framework`):
```bash
railway init
railway up
```

### Step 4: Generate a Public Domain
Once the build completes:
1. Go to your dashboard at [railway.app](https://railway.app).
2. Click on your project service &rarr; **Settings** tab.
3. Scroll down to **Networking** &rarr; Click **Generate Domain**.
4. Your application will be live at:
   `https://<your-service-name>.up.railway.app`

---

## Method 2: Deploy via GitHub (Continuous Deployment)

If you prefer linking your GitHub repository so Railway auto-deploys on every `git push`:

### Step 1: Upload `best_model.pt` to Cloud Storage
Upload `models/checkpoints/best_model.pt` (982MB) to one of the following:
* **Hugging Face Model Hub** (Free & unlimited for model weights)
* **GitHub Releases** (Supports files up to 2GB)
* **Google Drive / Dropbox / AWS S3** (Direct download link)

### Step 2: Connect GitHub in Railway
1. Go to [railway.app](https://railway.app) and click **New Project**.
2. Select **Deploy from GitHub repo**.
3. Choose your repository: `jayeshkondi-oss/deepfake-detection-framework`.

### Step 3: Add `MODEL_URL` Environment Variable
1. In your Railway project dashboard, click your service &rarr; **Variables** tab.
2. Add a new variable:
   * **Key**: `MODEL_URL`
   * **Value**: `<Direct_Download_Link_To_best_model.pt>`
3. When the container starts, `app.py` will automatically download `best_model.pt` into `models/checkpoints/` before running inference.

---

## Verification & Health Check

Once deployed, Railway monitors your container health using:
```
GET /api/health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "checkpoint_present": true
}
```

---

## Instant Alternative: Local Public Tunnel (1-Click)

If you want to immediately showcase the project to faculty without waiting for cloud builds:
Double-click `start_public_deployment.bat` in your project folder. It launches your local GPU-accelerated server and generates a public HTTPS link in 5 seconds.
