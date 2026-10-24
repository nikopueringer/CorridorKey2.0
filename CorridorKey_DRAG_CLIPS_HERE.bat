@echo off
rem CorridorKey 2.0 launcher. Drag a clip or folder onto this file, or
rem double-click it and drop the clip into the window when asked.
cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON=uv run"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if "%~1"=="" (
    %PYTHON% run_interactive.py
) else (
    %PYTHON% run_interactive.py "%~1"
)

echo.
pause
