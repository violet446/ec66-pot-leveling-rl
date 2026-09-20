@echo off
rem ============================================================
rem  Custom-size launcher for the Newton "mpm_granular" example.
rem
rem  Double-click this file, type the edge length of the particle
rem  block, press Enter -> the simulation window opens right away.
rem
rem  You can also run it with arguments from a terminal:
rem     run_block.bat 1.5
rem     run_block.bat 1.5 2.0           (edge 1.5 m, bottom at z=2.0)
rem     run_block.bat 1.5 --viewer null
rem
rem  FIXEDEDGE: put a number here (e.g. 1.5) to skip the question and
rem  always use that size.  Leave it empty to be asked every time.
rem ============================================================
set "FIXEDEDGE="

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

python C:\newton\run_block.py %FIXEDEDGE% %*
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
