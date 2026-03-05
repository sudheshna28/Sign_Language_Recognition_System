@echo off
echo ===================================================
echo     Starting ASL Sign Language Detector...
echo ===================================================

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. 
    echo Please run "setup.bat" first to install dependencies!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py

pause
