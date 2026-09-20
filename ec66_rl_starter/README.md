# EC66 + Newton 强化学习入门项目

本项目先跑通强化学习控制 EC66 的完整链路；最终目标为**甄锅内多次定体积铲取、抛回，保持料面平整**。甄锅外径 2 m，初始半锅不平物料，每铲暂定 2 L。详细参数和奖励设计见 [甄锅均料任务](docs/甄锅均料任务.md)。

第一版不是“学会挖沙”，而是让 EC66 在空场景中把末端执行器移动到一个随机目标点。这样可以把粒子、双向耦合、视觉、抓取和真机通信全部排除在外，只验证最关键的闭环：

```text
Newton 状态 -> observation -> PPO 策略 -> 关节目标增量 -> 关节位置 PD/drive -> Newton
                           ^                                      |
                           +------------- reward ----------------+
```

## 为什么这个目标合适

- 它沿用 EC66 的 URDF、关节限位、TCP 和关节驱动，不会绕开真实机械臂问题。
- 成功标准清楚：TCP 与目标点距离小于 3 cm。
- 训练失败时容易定位：观测、动作、奖励、重置和物理稳定性可以逐层检查。
- 后续增加料面高度图、装载量与动作阶段观测，并设计多次铲取抛回奖励；新策略需要重新训练。

## 先运行的两个命令

```powershell
cd C:\newton\ec66_rl_starter
python scripts\check_setup.py
python -m unittest discover -s tests -v
```

第一个命令确认 Newton、GPU 和 RL 训练依赖是否就绪；第二个命令验证最小任务的 observation 与 reward 约定。它们现在不训练策略，目的是让你先能看懂、能验证每个公式。

开始任何 Isaac Lab / RSL-RL 实验前，请先按 [启动 RL 实验说明](启动RL实验.md) 进入 Python 3.12 虚拟环境。

下载或等待安装期间，可按 [学习路径与资料](学习路径与资料.md) 建立强化学习和机械臂控制的必要基础。

EC66 的第一版采用 URDF 直接导入 Newton 的路线；有关为什么暂不需要 USD、以及后续代码如何分层，请看 [资产与实现路线](docs/资产与实现路线.md)。

## 当前进度（2026-09-19）

- **Gate 0 已通过**：`tests` 中 3 个任务契约测试全部通过。
- **Gate 1 已通过**：Cartpole Direct 已完成 PPO 训练、冻结策略可视化回放和策略导出；冻结策略能保持杆稳定。
- Gate 1 产物位于 `C:\newton\IsaacLab\logs\rsl_rl\cartpole_direct\2026-09-16_00-44-55`，其中包含 `model_49.pt`、`exported\policy.pt` 和 `exported\policy.onnx`。
- **Gate 2A 已通过**：单环境 EC66 刚体 smoke test 连续运行 250 个控制步，无 NaN、无关节越界或发散；小关节命令能产生预期关节响应和 TCP 位移。
- **Gate 2B 已通过**：固定可达目标、22 维 observation、6 维 action、reward、success 和 reset 闭环均已验证。
- **Gate 2C 已通过**：随机目标均由安全关节姿态经 FK 生成，初始距离覆盖 0.10–0.45 m；采样、reset 和可达性均已验证。
- **随机目标 PPO 已通过**：64 个并行环境训练 300 次更新、460,800 个交互样本；用新随机种子冻结评估 128/128 成功，平均终止距离 0.021031 m。
- **Gate 2D/2E 的 64 环境阶段已通过**：checkpoint、TorchScript、ONNX、严格评估入口和随机目标可视化回放均可用。
- **Gate 2F 已通过**：安全随机化起始关节姿态，并均衡训练 0.10–0.20 m、0.20–0.32 m、0.32–0.45 m 三个目标距离层；训练外冻结评估分别为 100/100、100/100、100/100。
- **Gate 3A 已通过**：2 m 外径空心甄锅、锅底/侧壁接触、机械臂保持和接触传感器正负测试已完成。当前仍用代理机械臂和铲子，未训练避碰或抛洒策略。
- **均料奖励接口已加入**：高度图起伏评价与体积/撒漏约束，4 项新增数学测试通过；尚未连接颗粒仿真。
- **Gate 3B 局部脚本动作已通过（2026-09-20）**：代理模型在三个局部点完成接近、下降、提升、返回；锅体接触为零，12 个阶段终点最大误差约 0.37 mm（仿真值）。控制方式是逆运动学 + 平滑轨迹 + 带有界积分修正的 PD，未训练新的 PPO。
- **真实模型演示入口已准备（2026-09-20）**：用户授权临时挂载；真实 EC66/铲子外观、重新计算的 TCP、135 块铲面碰撞已接入。机器人碰撞仍为代理，惯量未校准。三条局部动作通过，阶段终点最大误差约 0.56 mm（仿真值）。
- **下一步**：低分辨率颗粒静置、铲子接缝漏料与 2 L 装载量验证，再建立半锅不平物料和抛洒闭环。

## 文件地图

```text
ec66_rl_starter/
├─ README.md                         从这里开始
├─ docs/
│  ├─ roadmap.md                     总体阶段与每阶段的验收条件
│  ├─ task_contract.md               observation、action、reward 的逐项解释
│  └─ 资产与实现路线.md               URDF、版本边界与代码分层
├─ configs/
│  ├─ reach_v0.yaml                  Reach 任务契约配置
│  └─ pot_demo.json                  甄锅演示尺寸、位置与每铲体积
├─ src/ec66_rl/
│  ├─ reach_contract.py              可测试的任务数学：观测、动作、奖励
│  └─ newton_native/
│     ├─ scene_builder.py            固定基座 EC66 模型构建
│     ├─ reach_env.py                Newton 1.2.1 单环境与 PD drive
│     ├─ reach_task.py               固定目标任务闭环
│     ├─ batched_env.py              Newton 多 world GPU 环境
│     ├─ rsl_rl_env.py               RSL-RL VecEnv 适配器
│     ├─ train_reach.py              PPO 训练、评估和导出
│     ├─ evaluate_policy.py           独立随机目标冻结评估
│     ├─ play_policy.py              固定目标冻结策略回放
│     ├─ play_random_policy.py       随机目标冻结策略回放
│     └─ smoke_test.py               Gate 2A 自动验收入口
├─ tests/
│  └─ test_reach_contract.py         不依赖 GPU 的单元测试
└─ scripts/
   └─ check_setup.py                 当前机器的准备情况检查
```

## 学习与部署是两件事

训练阶段会同时运行许多个相同的 Newton 场景。每个场景反复执行“观察、动作、仿真一步、奖励、重置”；RSL-RL 的 PPO 根据这些经验更新神经网络。

部署阶段不再更新网络：每个控制周期只输入当前 observation，网络输出 action。实际接到机械臂前必须由限位、速度限制、力矩限制和急停组成独立安全层。

## 下一步顺序

Gate 0、Gate 1、Gate 2A–2F 均已完成。随机起点与随机目标的鲁棒策略位于：

```text
C:\newton\ec66_rl_starter\logs\rsl_rl\ec66_reach_robust\2026-09-17_20-42-12
```

在已设置环境变量并进入项目目录后，可启动冻结策略可视化回放：

```powershell
python -m ec66_rl.newton_native.play_random_policy --viewer gl --device cuda:0
```

红点是铲斗前缘 TCP，绿点是随机目标。命中后会短暂停留，再随机重置机械臂起始姿态并更换绿点。当前安全采样范围覆盖距离起点 0.10–0.45 m 的局部工作空间；下一步按 [roadmap](docs/roadmap.md) 进入 Gate 3 接触与简单物体。

RL 环境已具备 Python 3.12、PyTorch、RSL-RL、Isaac Lab 和 Newton 1.2.1。原 `ec66_sand` 工程使用系统 Python 3.13 / Newton 1.6.0；两套环境必须继续隔离，不能直接互相复制依赖特定版本的控制器代码。

`ec66_rl.newton_native` 会在导入 Newton/Warp 前，把当前进程的 `TEMP`、`TMP` 和 `WARP_CACHE_PATH` 自动指向 `C:\newton\warp_tmp` 与 `C:\newton\warp_cache`，避免中文 Windows 用户名导致 NVRTC 临时文件创建失败。

## 空心甄锅接触演示

```powershell
cd C:\newton\ec66_rl_starter
$env:PYTHONPATH = 'C:\newton\ec66_rl_starter\src'
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot --viewer gl --device cuda:0
```

画面是机械臂保持、两个测试球落入/撞击锅壁；没有粒子或新 PPO 动作。`configs/pot_demo.json` 除外径外均是可修改的演示参数。

## 锅内动作演示（Gate 3B）

```powershell
cd C:\newton\ec66_rl_starter
$env:PYTHONPATH = 'C:\newton\ec66_rl_starter\src'
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot_motion --viewer gl --device cuda:0
```

约 13 秒走完一次接近—下降—提升—返回，随后固定显示末态。红点为 TCP，绿点为当前阶段目标。可加 `--lateral-offset 0.05` 或 `--lateral-offset 0.10` 查看其他已验证局部点；负偏移曾触发锅沿碰撞，不属于已通过路径。全锅可达性、自碰撞、颗粒负载与实际安装参数尚未验证；临时真实外观/铲面碰撞版本的局部验证见下文。

临时真实模型使用额外选项：

```powershell
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_pot_motion --real-tool --viewer gl --device cuda:0
```

`--real-tool` 使用真实外观和拟合铲面碰撞，仍保留机器人简化碰撞。挂载假设在 `configs/real_tool_demo.json`，生成物在 `generated_assets/real_tool_demo`；修改配置后运行 `scripts/prepare_real_tool.py` 重新生成。原资产未修改，旧 PPO 回放仍使用原代理模型。模型拟合与回放使用 SciPy 1.18.1（本轮已补齐网格法线计算依赖），并非实测动力学标定。

## 静态铲面颗粒诊断

已增加独立 XPBD 粗颗粒测试，不是完整沙土任务或新 PPO。约 2 L 名义松装等效量，1326 颗中保留 1319 颗（99.47%），20 秒静置后通过落稳门槛；真实 2 L 容量、动态承料和物料参数尚未标定。详见 [颗粒铲面验收](docs/颗粒铲面验收.md)。

```powershell
cd C:\newton\ec66_rl_starter
$env:PYTHONPATH = 'C:\newton\ec66_rl_starter\src'
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.tool_particles --require-pass
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_tool_particles --viewer gl --device cuda:0
```

回放仅显示铲面碰撞块和颗粒，20 秒后停住。下一步先验证随铲运动与倾倒，再扩大到半锅物料和料面平整奖励。

## 真实铲子带料与倾倒（后续独立诊断）

现已实现真实铲子外观 + 默认可见颗粒 + 平移提升/倾倒/落稳。此入口仍不含整台 EC66，不是 PPO；先在后台静置装料，再显示动作，左侧显示阶段。详见 [带料倾倒验收](docs/带料倾倒验收.md)。

```powershell
cd C:\newton\ec66_rl_starter
$env:PYTHONPATH = 'C:\newton\ec66_rl_starter\src'
& 'C:\newton\env_isaaclab\Scripts\python.exe' -m ec66_rl.newton_native.play_tool_pour --viewer gl --device cuda:0
```

静置铲上 1319 颗，带料后 1318 颗；倾倒后全部落盘并落稳。落料高度图已接入，但观测不满整圆，暂不计算平整奖励。下一步将动作接回 EC66 与锅体约束。
