@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title elf3 - MuJoCo web viewer

REM ============================================================
REM  elf3 MuJoCo web viewer - Windows launcher
REM
REM  Double-click this file to start the local server and open
REM  the browser. Extra arguments are forwarded to server.mjs:
REM     start.bat --no-https
REM     start.bat --http-port 9000 --https-port 9443
REM
REM  !! KEEP THIS FILE ASCII-ONLY WITH CRLF LINE ENDINGS !!
REM  cmd.exe parses .bat byte-by-byte using the OEM codepage
REM  (CP936 on a Chinese Windows). UTF-8 Chinese text makes the
REM  parser swallow the following newline, which garbles every
REM  later line; bare LF endings break it too. Plain English
REM  comments + CRLF are the only safe combination.
REM ============================================================

cd /d "%~dp0"

REM ---------- locate node.exe: managed build first, then PATH ----------
set "NODE="
set "CAND=%USERPROFILE%\.workbuddy\binaries\node\versions\22.22.2-6\node.exe"
if exist "!CAND!" set "NODE=!CAND!"
if not defined NODE (
    where node >nul 2>nul
    if !errorlevel! equ 0 set "NODE=node"
)

if not defined NODE (
    echo.
    echo [ERROR] Node.js not found.
    echo         tried : !CAND!
    echo         tried : node in PATH
    echo         Install Node.js 18+ from https://nodejs.org/ and retry.
    echo.
    goto :fail
)

if not exist "%~dp0server.mjs" (
    echo.
    echo [ERROR] server.mjs not found in:
    echo         %~dp0
    echo         Keep this .bat inside the web\ folder.
    echo.
    goto :fail
)

echo.
echo ============================================================
echo   elf3 - MuJoCo web viewer / VR
echo   node   : !NODE!
echo   folder : %CD%
echo ============================================================
echo.
echo   Desktop browser : http://localhost:8080
echo   VR headset      : https://localhost:8443
echo.
echo   If a port is busy the server picks the next free one -
echo   the exact URLs are printed below.
echo   Press Ctrl+C to stop the server.
echo.

"!NODE!" server.mjs --open %*
set "RC=!errorlevel!"

echo.
if not "!RC!"=="0" (
    echo [ERROR] server.mjs exited with code !RC!
    goto :fail
)

echo [OK] server stopped normally.
echo Press any key to close this window...
pause >nul
exit /b 0

:fail
echo.
echo ------------------- FAILED -------------------
echo Press any key to close this window...
pause >nul
exit /b 1
