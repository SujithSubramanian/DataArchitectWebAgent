@echo off
REM Data Architect Agent - Windows Setup Script - Version 16
REM This script sets up the virtual environment and installs dependencies

echo ========================================
echo Data Architect Agent v16 - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found
python --version

REM Create virtual environment
echo.
echo [2/4] Creating virtual environment...
if exist dwebagent (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv dwebagent
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment and install dependencies
echo.
echo [3/4] Installing dependencies...
call dwebagent\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

REM CRITICAL FIX: Force correct httpx versions to avoid "proxies" error
echo.
echo [3.5/4] Fixing httpx compatibility (critical for Azure OpenAI)...
pip uninstall -y httpx httpcore
pip install httpx==0.24.1 httpcore==0.17.3 --no-deps
pip install httpx==0.24.1 httpcore==0.17.3

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [4/4] Setup complete!

REM Check if .env file exists
if not exist .env (
    echo.
    echo ========================================
    echo IMPORTANT: Azure OpenAI Configuration
    echo ========================================
    echo.
    echo A .env file has been created for you.
    echo.
    echo For Shell Azure OpenAI, edit .env and configure:
    echo   LLM_PROVIDER=azure
    echo   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    echo   AZURE_OPENAI_API_KEY=your-azure-api-key-here
    echo   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
    echo.
    echo See AZURE_OPENAI_SETUP.md for detailed instructions.
    echo Contact Shell IT if you need Azure OpenAI credentials.
    echo.
    
    REM Create .env template
    echo # Data Architect Agent Configuration > .env
    echo # > .env
    echo # For Shell Enterprise: Use Azure OpenAI >> .env
    echo LLM_PROVIDER=azure >> .env
    echo. >> .env
    echo # Azure OpenAI Configuration >> .env
    echo # Get these from Shell IT or Azure Portal >> .env
    echo AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/ >> .env
    echo AZURE_OPENAI_API_KEY=your-azure-api-key-here >> .env
    echo AZURE_OPENAI_API_VERSION=2024-02-15-preview >> .env
    echo AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4 >> .env
    echo. >> .env
    echo # Projects directory >> .env
    echo BASE_PATH=./projects >> .env
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Run: start.bat
echo.
pause
