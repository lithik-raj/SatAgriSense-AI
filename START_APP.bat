@echo off
echo ================================================================
echo   SENTINEL-EARTH AI Geo-Agricultural Platform v4.0
echo ================================================================
echo.
echo Starting unified server (Frontend + Backend + ML/DL + Database)...
start "Sentinel-Earth Server" cmd /k "cd /d "%~dp0backend" && python app.py"
echo.
echo Waiting for server to initialize...
timeout /t 3 /nobreak > nul
echo.
echo Launching Application at: http://localhost:5000
start "" "http://localhost:5000"
echo.
echo ================================================================
echo   URL: http://localhost:5000
echo   API: http://localhost:5000/api/status
echo ================================================================
pause
