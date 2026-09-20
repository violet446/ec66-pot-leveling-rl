# 启动 RL 实验前：进入 Isaac Lab 虚拟环境

所有 Isaac Lab、RSL-RL、PPO 训练、策略回放和策略导出实验，都必须在 `env_isaaclab` 虚拟环境中执行。

> 不要在这里运行原来的 `ec66_sand` phase 1–3 脚本。它们当前使用的是系统 Python 3.13；RL 实验使用独立的 Python 3.12 环境。

## 每次新开 PowerShell 后执行

```powershell
cd C:\newton
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env_isaaclab\Scripts\Activate.ps1
New-Item -ItemType Directory -Force C:\newton\warp_tmp, C:\newton\warp_cache | Out-Null
$env:TEMP = 'C:\newton\warp_tmp'
$env:TMP = 'C:\newton\warp_tmp'
$env:WARP_CACHE_PATH = 'C:\newton\warp_cache'
cd .\IsaacLab
```

成功后，PowerShell 提示符最前面会出现：

```text
(env_isaaclab) PS C:\newton\IsaacLab>
```

这时才可以执行 RL 相关命令。依赖现已安装；若需要修复或复核安装，可执行：

```powershell
.\isaaclab.bat -i
```

### 为什么需要设置 `TEMP`、`TMP` 和 `WARP_CACHE_PATH`

当前 Windows 用户目录含有中文字符。Warp 第一次编译 CUDA 内核时会调用 NVIDIA 的 NVRTC；它无法正确处理该目录中的中文路径，常见报错为 `invalid PCH directory` 或 `CUDA kernel build failed with error code 6`。

上面的三行把 Warp 的临时文件和缓存重定向到 `C:\newton` 下的纯英文目录。这是当前电脑运行 Newton / Isaac Lab CUDA 内核的必要设置；它只在当前 PowerShell 会话有效，因此每次新开 RL 终端都要执行。

## Gate 1 已完成

以下是已验证过的 Cartpole 训练命令，保留用于复现；进入 Gate 2 前不需要再次运行：

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task=Isaac-Cartpole-Direct-v0 --num_envs=256 physics=newton_mjwarp --visualizer newton
```

已验证产物：

```text
C:\newton\IsaacLab\logs\rsl_rl\cartpole_direct\2026-09-16_00-44-55
```

其中包含 `model_49.pt`、`exported\policy.pt`、`exported\policy.onnx`；冻结策略可视化回放能保持杆稳定。下一次实验应进入 Gate 2A 的单环境 EC66 smoke test，而不是继续训练 Cartpole。

## 验证当前是否处于正确环境

在 `C:\newton\IsaacLab` 内执行：

```powershell
python -c "import sys; print(sys.executable)"
```

输出应包含：

```text
C:\newton\env_isaaclab\Scripts\python.exe
```

安装完成后，也可以验证 RL 核心包和 GPU：

```powershell
python -c "import torch, newton, rsl_rl, isaaclab; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__); print('Newton:', newton.__version__)"
```

## 退出 RL 环境，运行原 EC66 沙土工程

```powershell
deactivate
cd C:\newton\ec66_sand
.\run_phase3.bat
```

也可以直接新开一个普通 PowerShell 窗口运行原 EC66 工程。这样原来的 Python 3.13 / Newton 1.6 环境不会受到 RL 依赖影响。
