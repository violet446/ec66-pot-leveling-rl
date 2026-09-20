"""Reach-v0 的纯数学部分。

本模块故意不导入 Newton、Isaac Lab 或 PPO。这样观测、动作和奖励可以在
CPU 单元测试中先被理解与锁定；后续 GPU 环境只负责提供真实状态和执行动作。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReachTaskConfig:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_joint_speed_rad_s: float = 2.0
    action_scale_rad: float = 0.04
    success_distance_m: float = 0.03
    progress_weight: float = 2.0
    distance_weight: float = -0.2
    action_l2_weight: float = -0.01
    success_bonus: float = 10.0
    failure_penalty: float = -5.0


def _vector(name: str, value: np.ndarray, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    return result


def build_observation(
    config: ReachTaskConfig,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    tcp_pos: np.ndarray,
    goal_pos: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Build the 22-element policy input in the documented, fixed order."""
    q = _vector("joint_pos", joint_pos, 6)
    qd = _vector("joint_vel", joint_vel, 6)
    tcp = _vector("tcp_pos", tcp_pos, 3)
    goal = _vector("goal_pos", goal_pos, 3)
    last_action = np.clip(_vector("previous_action", previous_action, 6), -1.0, 1.0)
    lower = _vector("config.joint_lower", config.joint_lower, 6)
    upper = _vector("config.joint_upper", config.joint_upper, 6)
    if np.any(upper <= lower):
        raise ValueError("Every joint upper limit must exceed its lower limit")
    q_normalized = 2.0 * (q - lower) / (upper - lower) - 1.0
    qd_normalized = np.clip(qd / config.max_joint_speed_rad_s, -1.0, 1.0)
    tcp_to_goal = goal - tcp
    distance = np.asarray([np.linalg.norm(tcp_to_goal)], dtype=np.float32)
    return np.concatenate((q_normalized, qd_normalized, tcp_to_goal, last_action, distance)).astype(np.float32)


def action_to_joint_target(config: ReachTaskConfig, joint_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Turn the policy's bounded output into a safe joint-position target."""
    q = _vector("joint_pos", joint_pos, 6)
    bounded_action = np.clip(_vector("action", action, 6), -1.0, 1.0)
    lower = _vector("config.joint_lower", config.joint_lower, 6)
    upper = _vector("config.joint_upper", config.joint_upper, 6)
    return np.clip(q + bounded_action * config.action_scale_rad, lower, upper).astype(np.float32)


def compute_reward(
    config: ReachTaskConfig,
    previous_distance_m: float,
    current_distance_m: float,
    action: np.ndarray,
    safety_failure: bool = False,
) -> tuple[float, bool, bool]:
    """Return ``(reward, terminated, succeeded)`` for one control step."""
    if previous_distance_m < 0.0 or current_distance_m < 0.0:
        raise ValueError("Distances must be non-negative")
    bounded_action = np.clip(_vector("action", action, 6), -1.0, 1.0)
    succeeded = current_distance_m <= config.success_distance_m
    reward = (
        config.progress_weight * (previous_distance_m - current_distance_m)
        + config.distance_weight * current_distance_m
        + config.action_l2_weight * float(np.mean(np.square(bounded_action)))
    )
    if succeeded:
        reward += config.success_bonus
    if safety_failure:
        reward += config.failure_penalty
    return float(reward), bool(succeeded or safety_failure), bool(succeeded)
