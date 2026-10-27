@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title elf3 - MuJoCo web viewer (motion replay)

REM ============================================================
REM  elf3 MuJoCo web viewer - one-click MOTION REPLAY mode
REM
REM  Difference from start.bat: this one opens the page already in
REM  "motion replay" mode (?motion=fight_policy) and makes sure the
REM  replay data exists before starting the server.
REM
REM  Usage: double-click it, or from a command line:
REM     start_vr_motion.bat --rebake         force re-record the replay data
REM     start_vr_motion.bat --http-port 9000 extra flags go to server.mjs
REM
REM  !! KEEP THIS FILE ASCII-ONLY WITH CRLF LINE ENDINGS !!
REM  See the note in start.bat - UTF-8 Chinese text or bare LF
REM  endings break cmd.exe parsing of this file.
REM ============================================================

cd /d "%~dp0"

REM ---------- split our own flags from the ones for server.mjs ----------
set "REBAKE="
set "ARGS="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--rebake" (
    set "REBAKE=1"
) else (
    set "ARGS=!ARGS! %~1"
)
shift
goto parse
:parsed

REM ---------- locate node.exe ----------
set "NODE="
set "CAND=%USERPROFILE%\.workbuddy\binaries\node\versions\22.22.2-6\node.exe"
if exist "!CAND!" set "NODE=!CAND!"
if not defined NODE (
    where node >nul 2>nul
    if !errorlevel! equ 0 set "NODE=node"
)
if not defined NODE (
    echo.
    echo [ERROR] Node.js not found. Install Node.js 18+ : https://nodejs.org/
    echo.
    goto :fail
)

REM ---------- locate python.exe (only needed to record the replay data) ----------
set "PY="
set "VENV=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "!VENV!" set "PY=!VENV!"
if not defined PY (
    where python >nul 2>nul
    if !errorlevel! equ 0 set "PY=python"
)

set "MOTION=assets\motion_fight_policy.bin"
if defined REBAKE goto bake
if not exist "%MOTION%" goto bake_none
goto run

:bake
if not defined PY (
    echo [NOTE] Python not found - keeping the existing replay data.
    goto run
)
echo.
echo [1/3] Re-recording motion replay data (MuJoCo + policy.onnx, 20-40 s) ...
echo.
"!PY!" "tools\bake_motion.py"
if errorlevel 1 echo [WARN] Recording failed - keeping the existing replay data.
goto run

:bake_none
echo.
echo [1/3] First run - recording motion replay data locally (MuJoCo + policy.onnx) ...
echo.
if defined PY (
    "!PY!" "tools\bake_motion.py"
    if errorlevel 1 echo [WARN] Recording failed - the page still opens but "motion replay" may fail to load.
) else (
    echo [WARN] Replay data assets\motion_fight_policy.bin is missing AND Python was not found.
    echo        The page still opens, but "motion replay" reports a load error.
    echo        To fix it: python web\tools\bake_motion.py
)

:run
echo.
echo [2/3] Starting the local server (HTTP + HTTPS) ...
echo [3/3] The browser opens directly in MOTION REPLAY mode.
echo       VR headset: same Wi-Fi, open the https:// LAN address printed below.
echo.
"!NODE!" server.mjs --open --open-path "/?motion=fight_policy"!ARGS!
echo.
echo Press any key to close this window...
pause >nul
exit /b 0

:fail
echo.
echo ------------------- FAILED -------------------
echo Press any key to close this window...
pause >nul
exit /b 1
