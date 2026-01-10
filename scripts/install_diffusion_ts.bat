@echo off
REM Installation script for Diffusion-TS model (Windows)
REM This script clones and installs Diffusion-TS from GitHub

echo ==================================
echo Diffusion-TS Installation Script
echo ==================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python first.
    exit /b 1
)
echo.

REM Check if Diffusion-TS is already installed
python -c "import diffusion_ts" 2>nul
if %errorlevel% == 0 (
    echo Diffusion-TS is already installed!
    echo.
    echo To reinstall, first uninstall:
    echo   pip uninstall diffusion-ts
    echo Then run this script again.
    exit /b 0
)

echo.
echo Diffusion-TS not found. Installing...
echo.

REM Create temp directory
set TEMP_DIR=%TEMP%\diffusion_ts_install
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"
echo Using temporary directory: %TEMP_DIR%

REM Clone repository
echo.
echo Cloning Diffusion-TS repository...
cd /d "%TEMP_DIR%"
git clone https://github.com/Y-debug-sys/Diffusion-TS.git
if errorlevel 1 (
    echo ERROR: Failed to clone repository. Check your internet connection and git installation.
    exit /b 1
)

REM Install package
echo.
echo Installing Diffusion-TS...
cd Diffusion-TS
pip install -e .
if errorlevel 1 (
    echo ERROR: Installation failed. Check the error messages above.
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import diffusion_ts; print('Diffusion-TS installed successfully!')"
if errorlevel 1 (
    echo ERROR: Installation verification failed.
    exit /b 1
)

echo.
echo ==================================
echo Installation Complete!
echo ==================================
echo.
echo You can now use the real Diffusion-TS model.
echo Test with:
echo   python -c "from src.models import IoTDiffusionGenerator; g = IoTDiffusionGenerator(); g.initialize(); print('Works!')"

REM Cleanup
echo.
echo Cleaning up temporary files...
cd /d %TEMP%
rmdir /s /q "%TEMP_DIR%"

echo.
echo Done!
