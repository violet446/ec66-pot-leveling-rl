# EC66 甄锅定体积铲取与抛洒均料

用 Newton 物理引擎 + 强化学习，让 EC66 机械臂在**圆锅（甄锅）**里反复铲取固定体积的颗粒物料并抛回锅内，在**物料总量不变**的前提下，尽量把高低不平的料面**摊平**。

- 初始状态：锅内约**半锅**物料，料面高低不平
- 每次动作：铲取固定体积（当前设计 **2 L**），抛回锅内另一位置
- 评价目标：每次抛洒落稳后重建锅内料面高度图，用**平整度**衡量本次动作的好坏
- 关键约束：不从锅外补料，因此"铲走凸起却不抛回"不能算改善

> 目标态的完整参数（锅体尺寸、每铲体积、奖励分量）见 [`ec66_rl_starter/docs/甄锅均料任务.md`](ec66_rl_starter/docs/甄锅均料任务.md)。

## 目录说明

```text
.
├─ ec66_rl_starter/          主项目：Newton + Isaac Lab + RSL-RL 的强化学习环境与验证脚本
│  ├─ README.md              主项目详细说明（从这份读起）
│  ├─ docs/                  任务契约、路线图、逐阶段验收记录
│  ├─ configs/               Reach 任务契约、甄锅演示尺寸、真实铲子挂载参数
│  ├─ src/ec66_rl/           核心代码（见下文）
│  ├─ scripts/               环境自检、真实资产检查与生成、绘图
│  ├─ tests/                 不依赖 GPU 的任务数学单元测试
│  ├─ generated_assets/      prepare_real_tool.py 生成的临时真实外观模型
│  └─ logs/                  验收报告与本机训练产物
├─ ec66_sand/                早期沙土工程（独立环境，见下方"两套环境"）
│  └─ assets/ec66_simplified.urdf   ← 主项目依赖此文件，勿删
└─ assets/                   原始资产（用户提供，未改动）
   ├─ ec66_description.urdf
   ├─ ec66/*.STL             EC66 base/link1–link6 网格
   └─ 铁锹4y向上.STL           真实铲子网格
```

⚠️ **必须保持这个目录结构**。代码里用的是相对定位，例如
`ec66_rl_starter/src/ec66_rl/newton_native/scene_builder.py` 中
`PROJECT_ROOT.parent / "ec66_sand" / "assets" / "ec66_simplified.urdf"`，
`scripts/prepare_real_tool.py` 中 `ROOT.parent / "assets"`。
如果把 `ec66_rl_starter` 单独拿出来，所有 URDF/STL 加载都会直接 `FileNotFoundError`。

## 未入库内容（需要自行准备）

仓库**不包含**下面这些体积巨大的目录，clone 后必须自己补：

| 目录 | 体积 | 说明 |
| --- | --- | --- |
| `env_isaaclab/` | 约 5.5 GB | Python 3.12 虚拟环境，RL 相关全部在这里跑 |
| `IsaacLab/` | 约 187 MB | 上游 [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) 的克隆 |
| `downloads/` | 约 2.7 GB | PyTorch 等离线安装包 |
| `warp_cache/`、`warp_tmp/` | — | Warp/NVRTC 运行期缓存，首次运行自动重建 |

依赖版本基线：完整快照见 [`requirements-lock.txt`](requirements-lock.txt)
（Python 3.12.11 / torch 2.10.0+cu128 / IsaacLab 3.0.0 @28a37cecd / Newton 1.2.1 /
warp-lang 1.13.0 / mujoco-warp 3.8.1 / rsl-rl-lib 5.0.1）。
另有 `scipy==1.18.1`（用于网格法线计算），记录在 [`ec66_rl_starter/requirements-assets.txt`](ec66_rl_starter/requirements-assets.txt)。

## 从零复现

### 最快方式：一条命令

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

脚本用 [uv](https://docs.astral.sh/uv/) 创建 Python 3.12 环境并装齐依赖，自动建好
`warp_tmp/`、`warp_cache/`，最后跑一次 import 自检并把下一步命令打印出来。

**默认不安装 IsaacLab**：`src/` 下没有任何 `import isaaclab`，本项目代码只依赖
`newton` / `warp` / `rsl_rl`，IsaacLab 只是文档里的启动外壳，以及 Gate 1 那个
Cartpole 内置任务的载体。跳过它可以省掉 187 MB 克隆和一整轮 `isaaclab.bat -i`。
确实需要 Isaac Lab 内置任务时加 `-Full`：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Full
```

> 脚本里的输出信息刻意只用 ASCII：Windows PowerShell 5.1 会把无 BOM 的 `.ps1`
> 按 ANSI 解析，中文字符串会显示成乱码。

下面是等价的**手工步骤**，脚本出问题时用来对照排查。

### 1. 拿到仓库

如果本仓库是 **Private**，需要先让仓库所有者把你加为协作者
（`Settings` → `Collaborators` → `Add people`），接受邀请后再克隆。
Public 仓库可跳过此步。

```powershell
cd D:\work                       # 建议放在纯 ASCII 路径下，见文末说明
git clone https://github.com/violet446/ec66-pot-leveling-rl.git
cd ec66-pot-leveling-rl
```

克隆目录名随意，仓库内部的 `ec66_rl_starter/`、`ec66_sand/`、`assets/` 相对结构由 Git 保证。

### 2.（可选）补齐 IsaacLab

只在需要 Isaac Lab 内置任务时才做。基线是 **IsaacLab 3.0.0 @ `28a37cecdd433c22d9eabd6a5954add9f13a8951`**，不要直接装最新版：

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 28a37cecdd433c22d9eabd6a5954add9f13a8951
cd ..
```

### 3. 建虚拟环境并安装依赖

基线用 **uv 0.12.14** 创建 Python 3.12.11 环境；用官方 venv 也可以：

```powershell
uv venv --python 3.12 env_isaaclab
# 没有 uv 就用：py -3.12 -m venv env_isaaclab

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env_isaaclab\Scripts\Activate.ps1

# Windows 上必须带 --index-url，否则从 PyPI 装到的是 CPU 版 torch
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

cd IsaacLab
.\isaaclab.bat -i
cd ..

python -m pip install -r requirements-lock.txt
```

### 4. 自检

```powershell
# 精简路径下没有 isaaclab，如果装了 -Full 可以再把它加回 import 列表
python -c "import torch,newton,rsl_rl; print('CUDA:', torch.cuda.is_available()); print('Torch:', torch.__version__); print('Newton:', newton.__version__)"

cd ec66_rl_starter
python scripts\check_setup.py
python -m unittest discover -s tests -v
```

### 5. 查看是否真正跑通

```powershell
$env:PYTHONPATH = "$PWD\src"
& '..\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot --viewer gl --device cuda:0
```

能弹出 GL 窗口、看到两个球落入甄锅即环境就绪。

### 关于路径里的中文

Windows 用户目录含中文字符时，Warp 首次编译 CUDA 内核会因 NVRTC 无法处理非 ASCII 路径而失败
（`invalid PCH directory` / `CUDA kernel build failed with error code 6`）。
`src/ec66_rl/newton_native/runtime_env.py` 会在导入 Newton/Warp 前把 `TEMP`、`TMP`、`WARP_CACHE_PATH`
重定向到仓库根下的 `warp_tmp/`、`warp_cache/`，所以只要**克隆位置本身不含中文**就能规避。
如果系统管理员用户名是中文，靠 `AppData\Local\Temp` 的默认路径仍会出问题——这也是上面三行环境变量的由来。

### 为什么不用容器（Docker）

评估过，在这个项目上容器**不会更快**，反而更慢，原因是：

1. **GL 可视化在容器里最麻烦**。`--viewer gl` 依赖 pyglet + OpenGL，Windows 上要经
   WSL2 + WSLg 转发；NVIDIA 驱动下的 GL 常常起不来。容器里基本只能跑 `--viewer null`
   的无窗口验收，交互式看料面形态还是得在宿主机上做。
2. **IsaacLab 官方的 `docker/` 没有覆盖 Newton 后端**。翻过它的 `Dockerfile.base` /
   `docker-compose.yaml`，全文 0 处提到 `newton`，那套镜像是围绕 Isaac Sim + PhysX/ovphysx 的。
   要用就得自己从 `nvidia/cuda:*-devel` 起（NVRTC 需要 devel 镜像）、装 Python 3.12、
   装全部依赖、再自己处理挂载与显卡透传，构建一次约 15–20 GB。
3. **它省不掉真正的成本**。5.5 GB 里大头是 torch 的 CUDA 运行时库，镜像里一样要装；
   而原机没有装 Docker，对方还得先装 Docker Desktop + WSL2，本身就是半小时起步。
4. **硬件门槛不变**。容器不改变"必须有 NVIDIA GPU 且驱动够新"这件事。
   本机基线是 RTX 4060 Laptop + 驱动 581.80；对方显存太小或没有 N 卡，容器也救不了。

**真正省时间的两个点**（已体现在 `setup_env.ps1` 里）：

- 用 **uv** 代替 pip，依赖解析与下载快一个数量级；原环境本来就是 uv 建的。
- **跳过 IsaacLab**：`src/` 下没有任何 `import isaaclab`，省掉 187 MB 克隆和一整轮
  `isaaclab.bat -i`。这是整套流程里最大的一块时间。

如果对方在一台**Linux 服务器**上、只需要无窗口训练（不需要 GL 回放），那么
容器是合理的，此时可以从 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 起，在里面照
`setup_env.ps1` 的同一套依赖装一遍即可——但那是另一个场景，不是"更快的复现"。

## 环境准备

RL 实验使用 **Python 3.12**（`env_isaaclab`）；`ec66_sand` 使用的是**系统 Python 3.13 / Newton 1.6.0**。两套环境必须隔离，不要互相复制依赖特定版本的控制器代码。

每个新开的 PowerShell 会话都要执行（完整说明见 [`ec66_rl_starter/启动RL实验.md`](ec66_rl_starter/启动RL实验.md)）：

```powershell
cd <仓库根>
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env_isaaclab\Scripts\Activate.ps1
New-Item -ItemType Directory -Force warp_tmp, warp_cache | Out-Null
$env:TEMP = "$PWD\warp_tmp"
$env:TMP = "$PWD\warp_tmp"
$env:WARP_CACHE_PATH = "$PWD\warp_cache"
```

为什么要这样设：Windows 用户目录含中文字符时，Warp 首次编译 CUDA 内核会因 NVRTC 无法处理非 ASCII 路径而失败（`invalid PCH directory` 或 `CUDA kernel build failed with error code 6`）。把临时目录和缓存重定向到纯 ASCII 路径即可。
`src/ec66_rl/newton_native/runtime_env.py` 会在导入 Newton/Warp 前自动完成同样的设置，直接跑 `python -m ec66_rl...` 时无需手动加。

自检：

```powershell
cd ec66_rl_starter
python scripts\check_setup.py
python -m unittest discover -s tests -v
```

## 快速开始

```powershell
cd ec66_rl_starter
$env:PYTHONPATH = "$PWD\src"

# 随机目标冻结策略回放（红点=TCP，绿点=目标）
& '..\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_random_policy --viewer gl --device cuda:0

# 空心甄锅接触演示
& '..\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot --viewer gl --device cuda:0

# 锅内动作演示（Gate 3B，约 13 秒一次接近—下降—提升—返回）
& '..\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot_motion --viewer gl --device cuda:0

# 铲面带料与倾倒诊断
& '..\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_tool_pour --viewer gl --device cuda:0
```

## 当前进度

已通过的验收（每项的详细数据见 `docs/` 下对应文档与 `logs/` 报告）：

- **Gate 1**：Cartpole Direct PPO 训练 → 冻结策略回放 → ONNX/TorchScript 导出
- **Gate 2A–2F**：单环境 EC66 刚体 smoke test、固定/随机目标闭环、随机起始关节姿态；64 并行环境训练 300 次更新，冻结评估三档目标距离各 100/100 成功
- **Gate 3A**：2 m 外径空心甄锅、锅底/侧壁接触传感器与故障测试
- **Gate 3B**：锅内局部动作脚本（逆运动学 + 平滑插值 + PD），三条局部路径无锅体接触，阶段终点误差约 0.37 mm
- **真实模型演示**：真实 EC66/铲子外观、重算 TCP、135 块铲面凹形碰撞（`--real-tool`），三条路径终点误差约 0.56 mm；机器人碰撞与惯量仍为代理
- **料面与奖励接口**：`src/ec66_rl/surface_metrics.py` 已实现顶部包络高度图、水平 RMS 起伏、P95–P05 高差、峰谷差与逐次抛洒奖励（7 项 CPU 测试通过）
- **颗粒诊断**：铲面静态装料约 2 L 名义等效量保留 99.47%，20 秒落稳；带料 1318 颗、倾倒后全部落盘

**尚未完成**：半锅不平物料的初始化、抛洒落料高度图覆盖整个圆盘、把 `surface_metrics` 接进真实粒子任务、多次铲取抛回的 PPO 训练、全锅可达性与避碰。

## 代码结构（`ec66_rl_starter/src/ec66_rl/`）

- `surface_metrics.py` — 料面高度图与平整度奖励（**本项目最终奖励的核心**）
- `reach_contract.py` — 不依赖 GPU 的任务数学：观测、动作、奖励
- `newton_native/`
  - `scene_builder.py` — 固定基座 EC66 模型构建
  - `pot_scene.py` / `play_pot.py` — 甄锅几何与接触验证
  - `pot_motion.py` / `play_pot_motion.py` — 锅内局部动作（Gate 3B）
  - `real_tool.py` — `--real-tool` 真实外观与铲面碰撞的挂载
  - `tool_particles.py` / `play_tool_particles.py` — 铲面静态颗粒诊断
  - `tool_pour.py` / `play_tool_pour.py` — 带料、倾倒与落料高度图
  - `reach_env.py` / `reach_task.py` / `batched_env.py` / `rsl_rl_env.py` — 单环境与多世界 GPU 环境、RSL-RL VecEnv 适配
  - `train_reach.py` / `evaluate_policy.py` / `play_*_policy.py` — 训练、评估与回放
  - `runtime_env.py` — 导入前重定向 Warp 临时目录与缓存

## 阅读顺序

1. 本文件
2. [`ec66_rl_starter/README.md`](ec66_rl_starter/README.md) — 主项目总览与命令
3. [`ec66_rl_starter/docs/roadmap.md`](ec66_rl_starter/docs/roadmap.md) — 阶段划分与验收条件
4. [`ec66_rl_starter/docs/甄锅均料任务.md`](ec66_rl_starter/docs/甄锅均料任务.md) — 目标、参数与奖励草案
5. [`ec66_rl_starter/docs/新手梳理-一步步理解本项目.md`](ec66_rl_starter/docs/新手梳理-一步步理解本项目.md) — 从零理解整体流程
6. [`ec66_rl_starter/docs/代码结构与全流程说明.md`](ec66_rl_starter/docs/代码结构与全流程说明.md)

## 协作约定

- 训练产生的中间 checkpoint（`logs/**/model_*.pt`）与 TensorBoard 事件文件不入库；`logs/**/exported/` 里的 `policy.pt` / `policy.onnx` 和验收报告、证据图保留。
- 未入库的大目录见上文表格，**不要**为了"方便"把 `env_isaaclab/`、`IsaacLab/`、`downloads/` 加进仓库。
- `assets/` 下的原始资产是用户提供且未改动的版本，不要就地修改；需要改生成模型请改 `configs/real_tool_demo.json` 后重跑 `scripts/prepare_real_tool.py`。
