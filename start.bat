@echo off
cd /d "%~dp0"

echo Local LLM Chat UI
echo ------------------------------------
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Creating virtual environment and installing dependencies...
    uv sync
) else (
    echo Virtual environment found.
)

echo.
uv run python app.py
pause
