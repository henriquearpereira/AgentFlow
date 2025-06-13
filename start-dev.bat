@echo off
echo ==========================================
echo   AI Research Agent - Development Server
echo ==========================================

:: Set the base directory
set "BASE_DIR=D:\code\AgentFlow"

:: Function to start backend
:start_backend
echo Starting Python Backend...
cd /d "%BASE_DIR%\pdF_research_agent"
if not exist "venv" (
    echo ERROR: Virtual environment not found. Please run setup-complete.bat first.
    pause
    exit /b 1
)

:: Start backend in a new window
start "AI Research Agent - Backend" cmd /k "venv\Scripts\activate.bat && python main.py"

:: Wait a moment for backend to start
timeout /t 3 /nobreak >nul

:: Function to start frontend
:start_frontend
echo Starting React Frontend...
cd /d "%BASE_DIR%\frontend"
if not exist "node_modules" (
    echo ERROR: Node modules not found. Please run setup-complete.bat first.
    pause
    exit /b 1
)

:: Start frontend in a new window
start "AI Research Agent - Frontend" cmd /k "npm start"

echo.
echo ==========================================
echo   Development Servers Started!
echo ==========================================
echo.
echo Backend (API):      http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Frontend (React):  http://localhost:3000
echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
echo The React app should automatically open in your browser.
echo If not, navigate to: http://localhost:3000
echo.
echo ==========================================

pause