@echo off
REM =============================================================================
REM CLV Prediction System - Windows Setup Script
REM Author: Ali Abbass (OTE22)
REM =============================================================================

echo.
echo  ====================================================
echo    CLV Prediction System - Setup Script
echo    Author: Ali Abbass (OTE22)
echo  ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [INFO] Python version: %PYVER%

REM Create virtual environment
echo.
echo [STEP 1/5] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo.
echo [STEP 2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [STEP 3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo [STEP 4/5] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed

REM Create necessary directories
echo.
echo [STEP 5/5] Creating directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
echo [SUCCESS] Directories created

REM Generate data if not exists
if not exist "data\customers.csv" (
    echo.
    echo [INFO] Generating sample data...
    python data\generate_data.py
)

echo.
echo  ====================================================
echo    Setup Complete!
echo  ====================================================
echo.
echo  To start the API server:
echo    1. Activate venv:  venv\Scripts\activate
echo    2. Run server:     python -m uvicorn backend.api_enhanced:app --reload
echo.
echo  Or use Docker:
echo    docker-compose up -d
echo.
echo  API will be available at: http://localhost:8000
echo  API Docs at: http://localhost:8000/api/docs
echo  Frontend at: http://localhost:3000 (with Docker)
echo.

pause
