@echo off
setlocal
cd /d "%~dp0"

:menu
cls
echo ====================================================
echo        Virtual Fashion Project Launcher
echo ====================================================
echo.
echo  [1] Start Project (Backend + Frontend)
echo  [2] Install Dependencies (Python + Node.js)
echo  [3] Start Backend Only
echo  [4] Start Frontend Only
 echo  [5] Exit
 echo  [6] Setup AI Models (Generate Embeddings)
 echo  [7] Migrate Data to Database (JSON -> DB)
 echo.
 set /p choice="Enter your choice (1-7): "

 if "%choice%"=="1" goto start_all
 if "%choice%"=="2" goto install
 if "%choice%"=="3" goto start_backend
 if "%choice%"=="4" goto start_frontend
 if "%choice%"=="5" goto exit
 if "%choice%"=="6" goto setup_ai
 if "%choice%"=="7" goto migrate_db
 echo Invalid choice. Please try again.
pause
goto menu

:start_all
echo.
echo Starting Backend and Frontend...
start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
start "Frontend App" cmd /k "cd frontend && npm run dev"
echo.
echo Both services are starting in new windows.
echo You can close this window now.
pause
goto exit

:install
echo.
echo Installing Backend Dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing backend dependencies.
    pause
    goto menu
)
cd ..

echo.
echo Installing Frontend Dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Error installing frontend dependencies.
    pause
    goto menu
)
cd ..

echo.
echo All dependencies installed successfully!
pause
goto menu

:start_backend
echo.
echo Starting Backend...
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
goto menu

:start_frontend
echo.
echo Starting Frontend...
cd frontend
npm run dev
pause
goto menu

:setup_ai
echo.
echo Setting up AI Models and Generating Embeddings...
cd backend
python setup_ai.py
cd ..
pause
goto menu

:migrate_db
echo.
echo Migrating Data to Database...
cd backend
python migrate_to_db.py
cd ..
pause
goto menu

:exit
endlocal
exit
