# Reach-v0：任务接口契约

这份文件比算法名称更重要。训练、回放和未来真机必须使用**完全相同**的观测顺序、单位和动作缩放。

## Observation（22 个浮点数）

| 段 | 宽度 | 内容 | 含义 |
| --- | ---: | --- | --- |
| `q_normalized` | 6 | 各关节角归一化到约 [-1, 1] | 机械臂姿态 |
| `qd_normalized` | 6 | 各关节速度除以最大安全速度 | 当前运动趋势 |
| `tcp_to_goal` | 3 | `goal_xyz - tcp_xyz`，单位 m | 还差多远、往哪走 |
| `previous_action` | 6 | 上一个 [-1, 1] 动作 | 让策略知道当前命令惯性 |
| `distance_to_goal` | 1 | TCP 到目标的欧氏距离，单位 m | 直接的任务进度 |

每个量都应该在 GPU 上产生并保持固定顺序。未来加入沙土时，新增内容只能追加到末尾，并提高配置中的 observation 维度；否则旧策略不可用。

## Action（6 个浮点数）

策略输出每个关节一个 `[-1, 1]` 数值。环境将它转换为小的目标位置变化：

```text
q_target = clamp(q_current + action * 0.04 rad, joint_lower, joint_upper)
```

`q_target` 交给关节位置 PD/joint drive。第一版不要求模型重力补偿，也不直接输出力矩；这是 Reach-v0 的重要简化与安全边界。

## Reward

每一步的奖励为：

```text
progress     = previous_distance - current_distance
reward       = 2.0 * progress
             - 0.2 * current_distance
             - 0.01 * mean(action^2)
             + 10.0  (仅当 current_distance <= 0.03 m)
             - 5.0   (发生碰撞或违反安全条件)
```

- `progress` 让策略每接近一点目标就得到提示；否则它可能需要偶然进入 3 cm 球才第一次得到奖励。
- 距离项让它在无进步时仍偏好靠近目标。
- 动作平方项偏好更平稳的控制。
- 成功与失败项使最终目标清楚。

系数只是第一版起点，不能当成物理常数。只有 Gate 2 的学习曲线不工作时才逐项调节，不能同时改 observation、动作范围和奖励系数。

## Reset 与终止

每回合最多 250 个控制步。成功、发生碰撞/安全违规或超时后都 reset。初始关节姿态在收紧后的安全关节限位内随机化。随机目标的第一版生成方式是：采样另一个安全关节姿态 `q_goal`，通过 FK 得到 `goal_xyz`，而不是直接在未经测量的笛卡尔长方体中采样。这样每个目标天然存在可行关节解，使策略学到“解决一类可达到达问题”，而不是记住单一轨迹或被不可达目标稀释奖励。
