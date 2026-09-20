@echo off
REM EC66 冻结策略可视化回放。
REM 直接用虚拟环境的 python.exe，不需要 Activate.ps1，因此不受 PowerShell 执行策略限制。
setlocal

if not exist C:\newton\warp_tmp mkdir C:\newton\warp_tmp
if not exist C:\newton\warp_cache mkdir C:\newton\warp_cache

set TEMP=C:\newton\warp_tmp
set TMP=C:\newton\warp_tmp
set WARP_CACHE_PATH=C:\newton\warp_cache
set PYTHONPATH=C:\newton\ec66_rl_starter\src

cd /d C:\newton\ec66_rl_starter

C:\newton\env_isaaclab\Scripts\python.exe -m ec66_rl.newton_native.play_policy --viewer gl --device cuda:0 %*

endlocal
