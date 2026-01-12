@echo off
REM Start ASL Recognition Project
REM This script activates the virtual environment and opens in the project directory

echo ================================================================================
echo ASL RECOGNITION SYSTEM - PROJECT STARTER
echo ================================================================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create it first:
    echo   py -3.11 -m venv venv
    echo.
    echo See INSTALL_PYTHON311.md for instructions
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [SUCCESS] Virtual environment activated!
echo [INFO] You should see (venv) in your prompt
echo.
echo ================================================================================
echo QUICK START COMMANDS
echo ================================================================================
echo.
echo Test system:
echo   python test_system_simple.py
echo.
echo Collect data:
echo   python src/data_collection/collect_static.py
echo   python src/data_collection/collect_dynamic.py
echo.
echo Train models:
echo   python src/training/train_static.py
echo   python src/training/train_dynamic.py
echo.
echo Run recognition:
echo   python src/inference/realtime_static.py
echo   python src/inference/realtime_dynamic.py
echo.
echo ================================================================================
echo.

REM Keep window open with activated environment
cmd /k
