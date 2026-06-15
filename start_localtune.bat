@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================
REM LocalTune Console - One-click Startup
REM Starts the local dashboard. Training can be launched from UI.
REM ============================================================

cd /d "%~dp0"

if not defined UV_LINK_MODE set "UV_LINK_MODE=copy"

echo ============================================================
echo   LocalTune Console
echo ============================================================
echo.

if not exist "pyproject.toml" (
    echo [ERROR] pyproject.toml not found.
    echo [ERROR] Please keep this file in the project root.
    pause
    exit /b 1
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if defined LOCALTUNE_HOST (
    set "HOST=%LOCALTUNE_HOST%"
) else (
    set "HOST=127.0.0.1"
)
if defined LOCALTUNE_PORT (
    set "PORT=%LOCALTUNE_PORT%"
) else (
    set "PORT=6543"
)
set "OPEN_BROWSER=1"
set "NO_TENSORBOARD="
set "NO_FRONTEND_BUILD="
set "SKIP_TRAINING_DEPS="
set "BROWSER_HOST="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--no-browser" (
    set "OPEN_BROWSER=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-tensorboard" (
    set "NO_TENSORBOARD=--no-tensorboard"
    shift
    goto parse_args
)
if /I "%~1"=="--no-frontend-build" (
    set "NO_FRONTEND_BUILD=--no-frontend-build"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-training-deps" (
    set "SKIP_TRAINING_DEPS=--skip-training-deps"
    shift
    goto parse_args
)
echo [WARN] Unknown argument ignored: %~1
shift
goto parse_args

:args_done
if not exist "%PYTHON_EXE%" (
    echo [WARN] Local venv Python not found: %PYTHON_EXE%
    echo [INFO] Falling back to: uv run python
    set "PYTHON_CMD=uv run python"
) else (
    set "PYTHON_CMD=%PYTHON_EXE%"
)

set "BROWSER_HOST=%HOST%"
if "%BROWSER_HOST%"=="0.0.0.0" set "BROWSER_HOST=127.0.0.1"
set "URL=http://%BROWSER_HOST%:%PORT%"

echo [INFO] Dashboard: %URL%
echo [INFO] TensorBoard is started by default. Use --no-tensorboard to skip it.
echo [INFO] Use --no-browser if you do not want the browser to open automatically.
echo.
echo [INFO] From LocalTune Console you can start smoke or formal training.
echo [INFO] Press Ctrl+C in this window to stop the dashboard.
echo.

if "%OPEN_BROWSER%"=="1" (
    echo [INFO] The browser will open after the dashboard is ready.
    start "" /b %PYTHON_CMD% scripts/wait_for_dashboard.py ^
        --health-url "%URL%/api/status" ^
        --browser-url "%URL%"
)

%PYTHON_CMD% scripts/start_dashboard.py --host %HOST% --port %PORT% %NO_TENSORBOARD% %NO_FRONTEND_BUILD% %SKIP_TRAINING_DEPS%

echo.
echo [INFO] Dashboard stopped.
pause
