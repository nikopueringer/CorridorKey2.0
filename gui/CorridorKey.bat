@echo off
rem CorridorKey 2.0 desktop UI. Double-click it, or drag clips or folders onto it.
cd /d "%~dp0.."

if not exist "venv\Scripts\pythonw.exe" (
    echo No venv found. Set up the environment as described in README.md first.
    pause
    exit /b 1
)

venv\Scripts\python.exe -c "import PySide6" 2>nul || (
    echo Installing the UI toolkit ^(PySide6^)...
    venv\Scripts\python.exe -m pip install PySide6-Essentials || (pause & exit /b 1)
)

start "" "venv\Scripts\pythonw.exe" -m gui %*
