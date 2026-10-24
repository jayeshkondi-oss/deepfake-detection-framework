# 100% Free Deployment: Hugging Face Spaces (16 GB RAM)

This guide shows you how to deploy your **DeepForensics AI** framework to **Hugging Face Spaces** for **100% FREE** with **16 GB of RAM** and a permanent public HTTPS URL.

---

## Why Hugging Face Spaces?
| Feature | Hugging Face Spaces | Render / Railway Free Tier |
| :--- | :--- | :--- |
| **Cost** | **100% Free Forever ($0)** | Paid after free trial / limited |
| **RAM (Memory)** | **16 GB RAM** (Runs ViT & Wav2Vec with ease) | 512 MB RAM (Crashes with Out-of-Memory) |
| **Model Size** | **Supports up to 50 GB** via Git LFS | Max 100 MB on GitHub |
| **Public URL** | Permanent `https://<user>-<space>.hf.space` | Temporary or sleep after inactivity |

---

## Step 1: Create a Free Hugging Face Space (30 seconds)
1. Go to [huggingface.co](https://huggingface.co) and sign up for a free account (or log in).
2. In the top right corner, click your profile picture &rarr; **New Space** (or go to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Fill in the details:
   * **Space name**: `deepfake-detector` (or any name you like)
   * **License**: `MIT`
   * **Space SDK**: Select **Docker** &rarr; Choose **Blank**
   * **Space hardware**: Select **Free &bull; 2 vCPU &bull; 16GB RAM**
   * **Visibility**: **Public**
4. Click **Create Space**.

---

## Step 2: Push Your Code & Model to Your Space

Hugging Face Spaces gives you a Git repository URL (e.g. `https://huggingface.co/spaces/<your-username>/deepfake-detector`).

Open **PowerShell** or **Command Prompt** in this project folder (`c:\Users\jayes\OneDrive\Desktop\deepfake-detection-framework`) and run:

```bash
# 1. Ensure Git LFS is active for the 982MB model weights
git lfs install
git lfs track "*.pt"
git add .gitattributes

# 2. Add your Hugging Face Space as a git remote
git remote add hf https://huggingface.co/spaces/<your-username>/deepfake-detector

# 3. Add the model checkpoint and push
git add models/checkpoints/best_model.pt
git add Dockerfile README.md app.py
git commit -m "deploy: initial hugging face spaces deployment"
git push hf main --force
```
*(When prompted for credentials, use your Hugging Face username and an **Access Token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

---

## Step 3: View Your Live App!

1. In your browser on Hugging Face, click the **App** tab.
2. You will see the container building (takes ~2 minutes).
3. Once the build finishes, your live deepfake detection dashboard will load directly on the page!
4. Anyone can access it at:
   `https://<your-username>-deepfake-detector.hf.space`

You can share this link with your faculty, professors, or add it to your project defense presentation!
