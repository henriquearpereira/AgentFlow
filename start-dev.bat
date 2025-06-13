@echo off
echo ==========================================
echo   AI Research Agent - Development Server
echo ==========================================

cd /d "D:\code\AgentFlow"
echo Current directory: %cd%

echo Starting Python Backend...
start "Backend" cmd /k "cd /d D:\code\AgentFlow\pdF_research_agent && D:\code\AgentFlow\venv\Scripts\activate.bat && python main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Checking frontend dependencies...
cd /d "D:\code\AgentFlow\frontend"
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
)

echo Starting React Frontend...
start "Frontend" cmd /k "cd /d D:\code\AgentFlow\frontend && npm start"

echo.
echo ==========================================
echo   Servers Started!
echo ==========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Both servers are running in separate windows.
echo ==========================================

pause