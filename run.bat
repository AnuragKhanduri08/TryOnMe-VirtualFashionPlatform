@echo off
echo ============================================
echo   AI Fashion Platform - Startup Script
echo ============================================

echo [1/2] Starting Backend Server...
start "Backend Server" cmd /k "cd backend && echo Starting FastAPI... && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo [2/2] Starting Frontend Application...
start "Frontend Application" cmd /k "cd frontend && echo Starting Next.js... && npm run dev"

echo.
echo ============================================
echo   SUCCESS! Project is running.
echo ============================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo.
echo   (Close the new terminal windows to stop)
echo ============================================
pause
