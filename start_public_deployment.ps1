# Deepfake Detection Framework - Public Deployment Launcher
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "      DEEPFAKE DETECTION FRAMEWORK: INSTANT PUBLIC DEPLOYMENT        " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Check if port 8000 is running
$activePort = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $activePort) {
    Write-Host "[*] Starting local DeepForensics server on port 8000..." -ForegroundColor Yellow
    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "app.py" -WindowStyle Hidden
    Start-Sleep -Seconds 6
} else {
    Write-Host "[+] Local DeepForensics server is active on port 8000." -ForegroundColor Green
}

# Fetch tunnel endpoint password IP
$tunnelIp = (curl.exe -s https://loca.lt/mytunnelpassword).Trim()

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "  PUBLIC DEPLOYMENT IS READY!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "[*] Your Public Endpoint IP: " -NoNewline
Write-Host $tunnelIp -ForegroundColor Yellow
Write-Host "[*] Note: If the browser asks for 'Endpoint IP / Password' on first visit, enter: $tunnelIp" -ForegroundColor DarkGray
Write-Host ""
Write-Host "[*] Launching public tunnel..." -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

npx localtunnel --port 8000
