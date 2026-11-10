@echo off
setlocal

rem ============================================================
rem  Prone crawl (双手双脚贴地爬行) - one-click launcher (Windows)
rem
rem  生成动作序列 -> 数值验收 -> 运动学回放窗口
rem
rem  双击即可；终端里可带参数：
rem      run_crawl.bat --no-replay        只生成 + 验收
rem      run_crawl.bat --phys             额外跑物理 PD 回放
rem      run_crawl.bat --speed 0.10 --cycles 6
rem
rem  换解释器：set PYTHON_EXE=C:\path\python.exe
rem ============================================================

set "PY=C:\Users\xx.mu\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if defined PYTHON_EXE set "PY=%PYTHON_EXE%"
if not exist "%PY%" (
    where py >nul 2>nul && set "PY=py"
)
if not exist "%PY%" set "PY=python"

set "ROOT=%~dp0"
set "LAUNCH=%ROOT%sim\vendor\bxi_controller_ros2\src\bxi_example_py_elf3\launch"
set "GEN=%LAUNCH%\prone_crawl.py"
set "REPLAY=%LAUNCH%\prone_replay.py"
set "CHECK=%ROOT%scripts\check_prone_crawl.py"

if not exist "%GEN%" (
    echo [ERROR] generator not found:
    echo         %GEN%
    goto :fail
)

echo Using python: %PY%
echo.

echo == 1/3 生成场景与轨迹 ==
"%PY%" "%GEN%" --write-scene %*
if not "%ERRORLEVEL%"=="0" goto :fail

echo.
echo == 2/3 数值验收 ==
"%PY%" "%CHECK%"
set "CRC=%ERRORLEVEL%"

echo.
echo == 3/3 运动学回放（关掉窗口结束）==
"%PY%" "%REPLAY%" --mode kin --loop

if not "%CRC%"=="0" (
    echo.
    echo [WARN] 数值验收未全项通过，详见 logs\prone_crawl\report.txt
    goto :fail
)

echo.
echo Done.
if "%~1"=="" pause
exit /b 0

:fail
if "%~1"=="" pause
exit /b 1
