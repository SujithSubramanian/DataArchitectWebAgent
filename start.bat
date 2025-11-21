@echo off
REM Data Architect Agent - Start Script - Version 16
REM This script starts the Flask backend server

echo ========================================
echo Data Architect Agent v16 - Starting Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist dwebagent (
    echo ERROR: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Check if .env exists and has API key
if not exist .env (
    echo ERROR: .env file not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

findstr /C:"your-azure-api-key-here" .env >nul
if %errorlevel% equ 0 (
    echo.
    echo WARNING: Azure OpenAI API key not configured
    echo Please edit .env and add your Azure credentials
    echo See AZURE_OPENAI_SETUP.md for instructions
    echo.
    pause
)

findstr /C:"your-api-key-here" .env >nul
if %errorlevel% equ 0 (
    echo.
    echo WARNING: OpenAI API key not configured
    echo Please edit .env and add your API key
    echo.
    pause
)

REM Activate virtual environment
call dwebagent\Scripts\activate.bat

REM Start the Flask server
echo.
echo Starting Data Architect Agent server...
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
