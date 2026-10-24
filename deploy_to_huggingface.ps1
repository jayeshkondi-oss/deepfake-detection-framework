# Deploy to Hugging Face Spaces (100% Free)
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "      DEEPFAKE DETECTION FRAMEWORK: HUGGING FACE SPACES DEPLOYER     " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a free Docker Space at: https://huggingface.co/new-space" -ForegroundColor Yellow
Write-Host "   (SDK: Docker -> Blank | Hardware: Free 16GB RAM)" -ForegroundColor DarkGray
Write-Host ""

$hfUsername = (Read-Host "Enter your Hugging Face Username").Trim()
$hfSpaceName = (Read-Host "Enter your Space Name (e.g. deepfake-detector)").Trim()

if (-not $hfUsername -or -not $hfSpaceName) {
    Write-Host "[!] Error: Username and Space Name are required." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[*] Initializing Git LFS for 982MB model weights..." -ForegroundColor Cyan
git lfs install
git lfs track "*.pt"
git add .gitattributes

Write-Host "[*] Configuring Hugging Face remote..." -ForegroundColor Cyan
git remote remove hf 2>$null
git remote add hf "https://huggingface.co/spaces/$hfUsername/$hfSpaceName"

Write-Host "[*] Staging files and model checkpoint..." -ForegroundColor Cyan
git add Dockerfile README.md app.py requirements.txt models/checkpoints/best_model.pt
git commit -m "deploy: deploy deepfake detection framework to hugging face spaces"

Write-Host ""
Write-Host "[*] Pushing code and weights to Hugging Face..." -ForegroundColor Yellow
Write-Host "[*] NOTE: When prompted for password, paste your Hugging Face Access Token:" -ForegroundColor Yellow
Write-Host "    https://huggingface.co/settings/tokens" -ForegroundColor DarkGray
Write-Host ""

git push hf main --force

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " Deployment push complete! Check your Space build at:" -ForegroundColor Green
Write-Host " https://huggingface.co/spaces/$hfUsername/$hfSpaceName" -ForegroundColor Yellow
Write-Host ""
Write-Host " Direct Public Web App URL:" -ForegroundColor Green
Write-Host " https://$hfUsername-$hfSpaceName.hf.space" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Green
