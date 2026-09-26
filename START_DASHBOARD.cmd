@echo off
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo Please run scripts\setup.ps1 first.
  pause
  exit /b 1
)
echo Open http://localhost:8501 in your browser.
".venv312\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
