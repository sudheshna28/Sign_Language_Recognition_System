@echo off
echo ===================================================
echo     ASL Sign Language Detector - Setup Script
echo ===================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python is installed.

:: Create virtual environment if it doesn't exist
IF NOT EXIST ".venv" (
    echo.
    echo [INFO] Creating virtual environment ".venv"...
    python -m venv .venv
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
python -m pip install --upgrade pip

echo [INFO] Installing required packages...
pip install -r requirements.txt
:: Also explicitly installing gTTS and deep-translator which are mentioned in README but might be missing from requirements.txt
pip install deep-translator gTTS

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install one or more dependencies.
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   Setup completed successfully!
echo ===================================================
echo.
echo To run the application, simply double-click on "run_app.bat"
echo.
pause
