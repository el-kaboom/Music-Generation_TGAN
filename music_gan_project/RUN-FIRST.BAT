@echo off
setlocal

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found! Please install Python first.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv music_gan_env

:: Activate virtual environment
echo Activating environment...
call music_gan_env\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

:: Install dependencies
echo Installing project dependencies...
pip install -r music_gan_project\requirements.txt

:: Keep the environment active
echo Virtual environment setup complete! 🎶
echo To activate the environment manually in the future, run:
echo call music_gan_env\Scripts\activate
cmd /k
