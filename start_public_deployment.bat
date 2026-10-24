@echo off
title Deepfake Detection Framework - Public Deployment
cls
echo =====================================================================
echo       DEEPFAKE DETECTION FRAMEWORK: INSTANT PUBLIC DEPLOYMENT
echo =====================================================================
echo.

:: Check if port 8000 is running
netstat -ano | findstr :8000 >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting local DeepForensics server on port 8000...
    start /b "" .venv\Scripts\python.exe app.py
    timeout /t 6 /nobreak >nul
) else (
    echo [+] Local DeepForensics server is already active on port 8000.
)

:: Get Endpoint IP for Localtunnel
for /f "tokens=*" %%a in ('curl -s https://loca.lt/mytunnelpassword') do set TUNNEL_IP=%%a

echo.
echo =====================================================================
echo   PUBLIC DEPLOYMENT IS READY!
echo =====================================================================
echo [*] Your Public Tunnel Password / Endpoint IP: %TUNNEL_IP%
echo [*] If prompted on first opening in browser, enter the IP above.
echo.
echo [*] Connecting secure public HTTPS tunnel...
echo =====================================================================
echo.

npx localtunnel --port 8000
pause
