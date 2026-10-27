@echo off
setlocal
title elf3 - motion tracking (MuJoCo)

REM ============ config ============
set "PYTHON=C:\Users\xx.mu\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
set "LAUNCH=%~dp0bxi_controller_ros2\src\bxi_example_py_elf3\launch"

set "POLICY=%~1"
set "SCENE=%~2"
if "%POLICY%"=="" set "POLICY=onnx"
if "%SCENE%"=="" set "SCENE=stairs"

echo ============================================================
echo   elf3 motion tracking   ^|   policy = %POLICY%   scene = %SCENE%
echo ============================================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] python.exe not found:
    echo         %PYTHON%
    echo         Edit PYTHON= at the top of this file.
    goto :fail
)

if not exist "%LAUNCH%\elf3_play.py" (
    echo [ERROR] elf3_play.py not found under:
    echo         %LAUNCH%
    goto :fail
)

cd /d "%LAUNCH%"
echo [INFO] working dir : %CD%
echo [INFO] launching MuJoCo native viewer ...
echo.

"%PYTHON%" elf3_play.py --policy %POLICY% --viewer native --scene %SCENE%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] elf3_play.py exited with code %RC%
    goto :fail
)

echo.
echo [OK] finished normally.
echo Press any key to close this window...
pause >nul
exit /b 0

:fail
echo.
echo ------------------- FAILED -------------------
echo Press any key to close this window...
pause >nul
exit /b 1
