@echo off
setlocal
cd /d "%~dp0"

set "TMPDIR=C:\warp_tmp"
set "TEMP=C:\warp_tmp"
set "TMP=C:\warp_tmp"
set "WARP_CACHE_PATH=C:\warp_cache"
set "NEWTON_CACHE_PATH=C:\newton\assets"

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%WARP_CACHE_PATH%" mkdir "%WARP_CACHE_PATH%"
if not exist "%NEWTON_CACHE_PATH%" mkdir "%NEWTON_CACHE_PATH%"

python run_ec66_phase3.py --viewer gl %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo [OK] EC66 phase 3 finished.) else (echo [FAIL] Exit code %RC%.)
pause
endlocal
