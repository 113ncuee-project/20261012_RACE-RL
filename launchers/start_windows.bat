@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Please create .venv and install the project as described in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m race_gui.server
if errorlevel 1 pause
