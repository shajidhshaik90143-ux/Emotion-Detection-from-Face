@echo off
echo ==========================================
echo EmotionVision AI - Setup
echo ==========================================
py -3.11 -m venv venv
if errorlevel 1 (
    echo Could not create Python 3.11 environment.
    echo Install Python 3.11 and try again.
    pause
    exit /b 1
)
call venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo.
echo Setup complete.
echo Run run.bat to start the application.
pause
