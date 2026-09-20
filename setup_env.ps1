<#
.SYNOPSIS
    One-command environment setup for the EC66 pot-leveling project (Windows + NVIDIA GPU).

.DESCRIPTION
    Creates a Python 3.12 virtual environment and installs everything the
    EC66 / pot workflow needs.

    Default (fast path, no IsaacLab):
        Covers pot_scene, pot_motion, tool_particles, tool_pour, smoke_test,
        train_reach, evaluate_policy and every play_* entry point.
        The project source under src/ never imports isaaclab; IsaacLab is only
        the launch shell for Isaac Lab built-in tasks.

    -Full:
        Additionally clones IsaacLab at the pinned commit and runs
        .\isaaclab.bat -i. Needed only for Isaac Lab built-in tasks
        (e.g. Isaac-Cartpole-Direct-v0, the Gate 1 milestone).

    NOTE: messages are intentionally ASCII-only. Windows PowerShell 5.1 reads
    .ps1 files without a BOM as ANSI, which would garble non-ASCII output.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Full
#>
[CmdletBinding()]
param(
    [switch]$Full,
    [string]$VenvDir         = 'env_isaaclab',
    [string]$PythonVersion   = '3.12',
    [string]$TorchVersion    = '2.10.0',
    [string]$TorchVisionVer  = '0.25.0',
    [string]$TorchIndex      = 'https://download.pytorch.org/whl/cu128',
    [string]$IsaacLabRepo    = 'https://github.com/isaac-sim/IsaacLab.git',
    [string]$IsaacLabCommit  = '28a37cecdd433c22d9eabd6a5954add9f13a8951'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if ([string]::IsNullOrEmpty($root)) { $root = (Get-Location).Path }
$venv = Join-Path $root $VenvDir
$py   = Join-Path $venv 'Scripts\python.exe'

function Write-Step { param([string]$Text) Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "  [OK]   $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "  [WARN] $Text" -ForegroundColor Yellow }
function Write-Info { param([string]$Text) Write-Host "         $Text" -ForegroundColor DarkGray }

function Invoke-Pip {
    param([string[]]$Arguments)
    if ($script:UseUv) {
        & uv pip install --python $py @Arguments
    } else {
        & $py -m pip install @Arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE)." }
}

# ------------------------------------------------------------------ 1/6 location
Write-Step '1/6  Repository location'
Write-Host "  root : $root"
Write-Host "  venv : $venv"
if ($root -match '[^\x00-\x7F]') {
    Write-Warn 'This path contains non-ASCII characters.'
    Write-Warn 'Warp / NVRTC may fail to compile CUDA kernels here'
    Write-Warn '("invalid PCH directory" or "CUDA kernel build failed with error code 6").'
    Write-Warn 'Move the clone to a pure ASCII path, e.g. D:\work\ec66-pot-leveling-rl, then rerun.'
}

# ------------------------------------------------------------------ 2/6 package manager
Write-Step '2/6  Package manager'
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
$UseUv = $null -ne $uvCmd
if ($UseUv) {
    Write-Ok "uv $(& uv --version)"
    Write-Info 'uv resolves and downloads roughly an order of magnitude faster than pip.'
} else {
    Write-Warn 'uv not found - falling back to python -m venv + pip, which is much slower.'
    Write-Warn 'Install it with:  winget install astral-sh.uv'
}

# ------------------------------------------------------------------ 3/6 interpreter
Write-Step "3/6  Virtual environment (Python $PythonVersion)"
if (Test-Path $py) {
    Write-Ok "Reusing existing environment: $venv"
} else {
    if ($UseUv) {
        & uv venv --python $PythonVersion $venv
        if ($LASTEXITCODE -ne 0) { throw 'uv venv failed.' }
    } else {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $pyLauncher) {
            & py "-$PythonVersion" -m venv $venv
        } else {
            & python -m venv $venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'python -m venv failed.' }
    }
    Write-Ok "Created $venv"
}

$actual = (& $py -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))")
if ($actual -eq $PythonVersion) {
    Write-Ok "Interpreter version $actual"
} else {
    Write-Warn "Interpreter version is $actual, expected $PythonVersion."
    Write-Warn 'The project baseline is 3.12; other versions are untested.'
}

# ------------------------------------------------------------------ 4/6 torch
Write-Step "4/6  PyTorch $TorchVersion (CUDA 12.8 build)"
Write-Info "index: $TorchIndex"
Write-Info 'Must be installed from the CUDA index: on Windows the PyPI wheel is CPU-only.'
Invoke-Pip @('--index-url', $TorchIndex, "torch==$TorchVersion", "torchvision==$TorchVisionVer")
Write-Ok 'PyTorch installed'

# ------------------------------------------------------------------ 5/6 project deps
Write-Step '5/6  Project dependencies'
$minReq = Join-Path $root 'requirements-min.txt'
if (Test-Path $minReq) {
    Invoke-Pip @('-r', $minReq)
    Write-Ok "Installed from requirements-min.txt"
} else {
    Write-Warn 'requirements-min.txt not found; installing the equivalent set inline.'
    Invoke-Pip @('newton[examples]==1.2.1', 'numpy==2.5.3', 'scipy==1.18.1', 'rsl-rl-lib==5.0.1')
    Write-Ok 'Core dependencies installed'
}
Write-Info 'Includes newton, warp-lang, mujoco, mujoco-warp, pyglet and imgui-bundle.'

# ------------------------------------------------------------------ 6/6 IsaacLab
Write-Step '6/6  IsaacLab'
if ($Full) {
    $lab = Join-Path $root 'IsaacLab'
    if (Test-Path (Join-Path $lab '.git')) {
        Write-Ok "Existing clone found: $lab"
        & git -C $lab fetch --all --tags
    } else {
        & git clone $IsaacLabRepo $lab
        if ($LASTEXITCODE -ne 0) { throw 'git clone of IsaacLab failed.' }
    }
    & git -C $lab checkout $IsaacLabCommit
    if ($LASTEXITCODE -ne 0) { throw "git checkout $IsaacLabCommit failed." }
    Write-Ok "Checked out $IsaacLabCommit"

    # isaaclab.bat resolves the interpreter from VIRTUAL_ENV, so set it explicitly
    # instead of requiring an activated shell.
    $env:VIRTUAL_ENV = $venv
    $env:PATH = (Join-Path $venv 'Scripts') + ';' + $env:PATH
    Push-Location $lab
    try {
        & (Join-Path $lab 'isaaclab.bat') -i
        if ($LASTEXITCODE -ne 0) { throw 'isaaclab.bat -i failed.' }
    } finally {
        Pop-Location
    }
    Write-Ok 'IsaacLab installed'
} else {
    Write-Info 'Skipped (fast path). Add -Full if you need Isaac Lab built-in tasks'
    Write-Info 'such as Isaac-Cartpole-Direct-v0. Nothing in src/ imports isaaclab.'
}

# ------------------------------------------------------------------ caches
New-Item -ItemType Directory -Force (Join-Path $root 'warp_tmp')   | Out-Null
New-Item -ItemType Directory -Force (Join-Path $root 'warp_cache') | Out-Null

# ------------------------------------------------------------------ verify
Write-Step 'Verification'
& $py -c "import torch, newton, warp, rsl_rl, numpy, scipy; print('  torch ', torch.__version__); print('  cuda  ', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('  newton', newton.__version__); print('  warp  ', warp.__version__); print('  numpy ', numpy.__version__); print('  scipy ', scipy.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Warn 'Import check failed - see the traceback above.'
} else {
    Write-Ok 'All core imports succeeded'
}

Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Cyan
Write-Host "  cd $root\ec66_rl_starter"
Write-Host "  `$env:PYTHONPATH = `"`$PWD\src`""
Write-Host "  & '..\$VenvDir\Scripts\python.exe' -m ec66_rl.newton_native.pot_motion      # headless acceptance"
Write-Host "  & '..\$VenvDir\Scripts\python.exe' -m ec66_rl.newton_native.play_pot --viewer gl --device cuda:0"
Write-Host ''
Write-Host "  python scripts\check_setup.py" -ForegroundColor DarkGray
Write-Host "  python -m unittest discover -s tests -v" -ForegroundColor DarkGray
