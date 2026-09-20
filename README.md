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

额外依赖：`scipy==1.18.1`（用于网格法线计算），见 [`ec66_rl_starter/requirements-assets.txt`](ec66_rl_starter/requirements-assets.txt)。

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
