@echo off
title Deploy to Hugging Face Spaces (100% Free)
cls
echo =====================================================================
echo       DEEPFAKE DETECTION FRAMEWORK: HUGGING FACE SPACES DEPLOYER
echo =====================================================================
echo.
echo 1. Create a free Docker Space at: https://huggingface.co/new-space
echo    (SDK: Docker -> Blank, Hardware: Free 16GB RAM)
echo.
set /p HF_USERNAME="Enter your Hugging Face Username: "
set /p HF_SPACE_NAME="Enter your Space Name (e.g. deepfake-detector): "

echo.
echo [*] Initializing Git LFS for 982MB model weights...
git lfs install
git lfs track "*.pt"
git add .gitattributes

echo [*] Configuring Hugging Face remote...
git remote remove hf 2>nul
git remote add hf https://huggingface.co/spaces/%HF_USERNAME%/%HF_SPACE_NAME%

echo [*] Staging files and model checkpoint...
git add Dockerfile README.md app.py requirements.txt models/checkpoints/best_model.pt
git commit -m "deploy: deploy deepfake detection framework to hugging face spaces"

echo.
echo [*] Pushing code and weights to Hugging Face...
echo [*] NOTE: When prompted for password, paste your Hugging Face Access Token:
echo     https://huggingface.co/settings/tokens
echo.
git push hf main --force

echo.
echo =====================================================================
echo Deployment push complete! Check your Space build at:
echo https://huggingface.co/spaces/%HF_USERNAME%/%HF_SPACE_NAME%
echo.
echo Direct Public Web App URL:
echo https://%HF_USERNAME%-%HF_SPACE_NAME%.hf.space
echo =====================================================================
pause
