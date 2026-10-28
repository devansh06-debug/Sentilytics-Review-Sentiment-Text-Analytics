@echo off
title Customer Review Sentiment Analysis
cd /d "%~dp0"

echo ========================================================
echo   Customer Review Sentiment Analysis Project
echo ========================================================
echo.

:: 1. Ensure virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment...
    python -m venv .venv
    echo [2/3] Installing required packages...
    .venv\Scripts\pip.exe install -r requirements.txt
) else (
    echo [OK] Virtual environment ready.
)

:: 2. Ensure pipeline outputs and model exist
if not exist "outputs\lstm_model.pt" (
    echo [3/3] Running data pipeline and training models...
    .venv\Scripts\python.exe run_all.py
) else (
    echo [OK] Trained models and data found in outputs/
)

echo.
echo ========================================================
echo   Launching Streamlit Web App (http://localhost:8501)
echo ========================================================
echo Press Ctrl+C in this window anytime to stop the server.
echo.

.venv\Scripts\streamlit.exe run app.py --browser.gatherUsageStats false

pause
