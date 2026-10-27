@echo off
setlocal
title elf3 - climb stairs (MuJoCo)

REM ============ config ============
set "PYTHON=C:\Users\xx.mu\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
set "LAUNCH=%~dp0bxi_controller_ros2\src\bxi_example_py_elf3\launch"

set "POLICY=%~1"
set "DUR=%~2"
if "%POLICY%"=="" set "POLICY=normal_depth"
if "%DUR%"=="" set "DUR=40"

echo ============================================================
echo   elf3 climb stairs   ^|   policy = %POLICY%   duration = %DUR%s
echo ============================================================
echo.

if not exist "%PYTHON%" (
    echo [ERROR] python.exe not found:
    echo         %PYTHON%
    echo         Edit PYTHON= at the top of this file.
    goto :fail
)

if not exist "%LAUNCH%\climb_stairs.py" (
    echo [ERROR] climb_stairs.py not found under:
    echo         %LAUNCH%
    goto :fail
)

cd /d "%LAUNCH%"
echo [INFO] working dir : %CD%
echo [INFO] launching MuJoCo native viewer ...
echo.

"%PYTHON%" climb_stairs.py --policy %POLICY% --duration %DUR% --stand-time 2 --cmd-vx 0.5 --viewer native
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] climb_stairs.py exited with code %RC%
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
