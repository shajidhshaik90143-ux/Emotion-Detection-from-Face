@echo off
echo ==========================================
echo EmotionVision AI - Starting
echo ==========================================
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate
streamlit run app.py
pause
