@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo   AI Research Agent - Complete Setup
echo ==========================================

:: Set the base directory (AgentFlow)
set "BASE_DIR=D:\code\AgentFlow"
cd /d "%BASE_DIR%"

echo Current directory: %cd%

:: Check if Node.js is installed
echo.
echo [1/6] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    echo After installing Node.js, run this script again.
    pause
    exit /b 1
)

echo ✓ Node.js is installed
node --version

:: Check if npm is available
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: npm is not available
    pause
    exit /b 1
)

echo ✓ npm is available
npm --version

:: Create frontend directory and setup React app
echo.
echo [2/6] Setting up React frontend...

if exist "frontend" (
    echo Frontend directory already exists. Do you want to remove and recreate it? (y/n)
    set /p choice=
    if /i "!choice!"=="y" (
        echo Removing existing frontend directory...
        rmdir /s /q frontend
    ) else (
        echo Keeping existing frontend directory.
        goto :setup_backend
    )
)

:: Create React app with TypeScript
echo Creating React app with TypeScript...
npx create-react-app frontend --template typescript
if %errorlevel% neq 0 (
    echo ERROR: Failed to create React app
    pause
    exit /b 1
)

echo ✓ React app created successfully

:: Navigate to frontend directory
cd frontend

:: Install additional dependencies
echo.
echo [3/6] Installing additional dependencies...
npm install lucide-react tailwindcss autoprefixer postcss
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed

:: Initialize Tailwind CSS
echo.
echo [4/6] Setting up Tailwind CSS...
npx tailwindcss init -p
if %errorlevel% neq 0 (
    echo ERROR: Failed to initialize Tailwind CSS
    pause
    exit /b 1
)

echo ✓ Tailwind CSS initialized

:: Copy configuration files (you'll need to manually replace these)
echo.
echo [5/6] Setting up configuration files...
echo Please manually replace the following files with the provided content:
echo   - src/App.tsx
echo   - src/App.css
echo   - tailwind.config.js
echo   - package.json (merge dependencies)
echo.
echo The files are ready in the artifacts above.

:: Navigate back to base directory
cd /d "%BASE_DIR%"

:setup_backend
:: Setup Python virtual environment for backend
echo.
echo [6/6] Setting up Python backend environment...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/
    pause
    exit /b 1
)

echo ✓ Python is available

:: Navigate to the backend directory
cd /d "%BASE_DIR%\pdF_research_agent"

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

:: Activate virtual environment and install requirements
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install fastapi uvicorn python-dotenv websockets python-multipart
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

echo ✓ Python dependencies installed

:: Create a sample .env file
if not exist ".env" (
    echo Creating sample .env file...
    echo # AI Research Agent Environment Variables > .env
    echo # Add your API keys here >> .env
    echo # OPENAI_API_KEY=your_openai_key_here >> .env
    echo # ANTHROPIC_API_KEY=your_anthropic_key_here >> .env
    echo # GOOGLE_API_KEY=your_google_key_here >> .env
    echo # GROQ_API_KEY=your_groq_key_here >> .env
    echo ✓ Sample .env file created
)

:: Create reports directory
if not exist "reports" (
    mkdir reports
    echo ✓ Reports directory created
)

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. FRONTEND SETUP:
echo    - Navigate to D:\code\AgentFlow\frontend
echo    - Replace the following files with provided content:
echo      * src/App.tsx
echo      * src/App.css  
echo      * tailwind.config.js
echo    - Run: npm start
echo    - Frontend will be available at: http://localhost:3000
echo.
echo 2. BACKEND SETUP:
echo    - Navigate to D:\code\AgentFlow\pdF_research_agent
echo    - Activate venv: venv\Scripts\activate.bat
echo    - Add your API keys to .env file
echo    - Run: python main.py
echo    - Backend API will be available at: http://localhost:8000
echo    - API docs at: http://localhost:8000/docs
echo.
echo 3. USAGE:
echo    - Start backend first (Python)
echo    - Start frontend second (React)
echo    - Open browser to http://localhost:3000
echo    - Enter research questions and generate reports!
echo.
echo ==========================================
pause