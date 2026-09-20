@echo off
rem ============================================================
rem  One-shot launcher for NVIDIA Newton / Warp examples.
rem  WHY: the Windows profile path contains non-ASCII characters
rem  (C:\Users\<chinese-name>), which breaks the NVRTC/EDG
rem  temporary files, the CUDA precompiled-header (PCH) folder,
rem  the Warp kernel cache path and the Newton asset cache.
rem  This script points every temp/cache path to plain-ASCII
rem  folders FOR THIS CMD SESSION ONLY. It never calls setx, so
rem  nothing is written to the system environment and everything
rem  is gone as soon as the window is closed.
rem
rem  Usage:
rem    run_newton.bat                              (asks for a name)
rem    run_newton.bat mpm_granular                 (opens the GUI)
rem    run_newton.bat mpm_granular --viewer null --benchmark 5
rem  Any number of extra arguments is forwarded to the example.
rem ============================================================
setlocal
cd /d "%~dp0"

set "TMPDIR=C:\warp_tmp"
set "TEMP=C:\warp_tmp"
set "TMP=C:\warp_tmp"
set "WARP_CACHE_PATH=C:\warp_cache"
set "NEWTON_CACHE_PATH=C:\newton\assets"

if not exist "%TEMP%"              mkdir "%TEMP%"
if not exist "%WARP_CACHE_PATH%"   mkdir "%WARP_CACHE_PATH%"
if not exist "%NEWTON_CACHE_PATH%" mkdir "%NEWTON_CACHE_PATH%"

rem ------- which example to run ---------------------------------
set "EXAMPLE=%~1"
if "%EXAMPLE%"=="" set /p "EXAMPLE=Example name [mpm_granular]: "
if "%EXAMPLE%"=="" set "EXAMPLE=mpm_granular"

rem ------- collect ALL remaining arguments (no 8-arg limit) ----
set "EXTRA="
:collect_args
shift
if "%~1"=="" goto args_collected
set "EXTRA=%EXTRA% %~1"
goto collect_args
:args_collected

echo.
echo ============================================================
echo  example          : %EXAMPLE%
echo  extra args       :%EXTRA%
echo  TMPDIR/TEMP/TMP  : %TEMP%
echo  WARP_CACHE_PATH  : %WARP_CACHE_PATH%
echo  NEWTON_CACHE_PATH: %NEWTON_CACHE_PATH%
echo ============================================================
echo.

python -m newton.examples "%EXAMPLE%" %EXTRA%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [OK] finished successfully ^(exit code 0^).
) else (
    echo [FAIL] python exited with code %RC%.
)
echo.
pause
endlocal
