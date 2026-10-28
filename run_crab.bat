@echo off
setlocal

rem ============================================================
rem  Crab-walk stair climbing - one-click launcher (Windows)
rem
rem  Double-click this file to run the full pipeline:
rem      terrain -> trajectory -> MuJoCo replay window
rem
rem  From a terminal you can pass arguments, e.g.
rem      run_crab.bat --preset easy
rem      run_crab.bat --seed 7 --step-time 1.2
rem      run_crab.bat --show-config
rem      run_crab.bat --stages gen,plan
rem
rem  To use another interpreter:  set PYTHON_EXE=C:\path\python.exe
rem ============================================================

rem --- 1. Python interpreter -------------------------------------------------
set "PY=C:\Users\xx.mu\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if defined PYTHON_EXE set "PY=%PYTHON_EXE%"
if not exist "%PY%" (
    where py >nul 2>nul && set "PY=py"
)
if not exist "%PY%" set "PY=python"

rem --- 2. Paths --------------------------------------------------------------
set "ROOT=%~dp0"
set "RUNNER=%ROOT%sim\vendor\bxi_controller_ros2\src\bxi_example_py_elf3\launch\run_crab.py"

if not exist "%RUNNER%" (
    echo [ERROR] runner not found:
    echo         %RUNNER%
    goto :fail
)

echo Using python: %PY%
echo.

"%PY%" "%RUNNER%" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [FAILED] exit code %RC%
    goto :fail
)

echo.
echo Done.
rem Pause only when double-clicked (no arguments) so terminal use stays scriptable.
if "%~1"=="" pause
exit /b 0

:fail
if "%~1"=="" pause
exit /b 1
