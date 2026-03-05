@echo off
echo ===================================================
echo     ASL Sign Language Detector - Setup Script
echo ===================================================
echo.

:: Check if Python is installed
set PYTHON_CMD=python
%PYTHON_CMD% --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    set PYTHON_CMD=python3
    %PYTHON_CMD% --version >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python is not installed or not in your PATH.
        echo Please install Python 3.10 or 3.11 from https://www.python.org/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
)

echo [OK] Using %PYTHON_CMD%...

:: Create virtual environment if it doesn't exist
IF NOT EXIST ".venv" (
    echo.
    echo [INFO] Creating virtual environment ".venv"...
    %PYTHON_CMD% -m venv .venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) ELSE (
    echo [OK] Virtual environment ".venv" already exists.
)

:: Activate virtual environment and install dependencies
echo.
echo [INFO] Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip

echo [INFO] Installing required packages...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   Setup completed successfully!
echo ===================================================
echo.
echo [IMPORTANT] If you see a "Model Loading" error later:
echo 1. Pull the latest code (git pull origin main)
echo 2. Run: python train_landmarks.py
echo This will regenerate the model for your specific computer.
echo.
pause
