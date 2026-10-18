@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    py -3.11 -m venv venv
    if errorlevel 1 goto :error
)

"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup completed. Double-click run_app.bat to start the detector.
pause
exit /b 0

:error
echo.
echo Setup failed. Check the error above and confirm Python 3.11 is installed.
pause
exit /b 1
