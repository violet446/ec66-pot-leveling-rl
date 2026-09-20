"""Single-environment fixed-goal Reach-v0 task used for Gate 2B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ec66_rl.reach_contract import ReachTaskConfig, build_observation, compute_reward

from .reach_env import Ec66ReachEnv
from .scene_builder import INITIAL_JOINT_POS


EPISODE_LENGTH_STEPS = 250

# A nearby, legal pose keeps Gate 2B focused on validating the task loop.  Gate
# 2C will replace this with joint-space sampling and FK-generated random goals.
FIXED_GOAL_JOINT_POS = INITIAL_JOINT_POS + np.array(
    [0.18, -0.12, 0.10, 0.0, 0.0, 0.0], dtype=np.float32
)


@dataclass(frozen=True)
class ReachStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    succeeded: bool
    distance_m: float


class FixedGoalReachTask:
    """Connect physics state to the documented Reach-v0 task contract."""

    def __init__(self, *, device: str = "cuda:0") -> None:
        self.env = Ec66ReachEnv(device=device)
        self.config = ReachTaskConfig(
            joint_lower=self.env.joint_lower,
            joint_upper=self.env.joint_upper,
        )
        self.goal_joint_pos = np.clip(
            FIXED_GOAL_JOINT_POS,
            self.env.joint_lower,
            self.env.joint_upper,
        ).astype(np.float32)
        self.goal_position = self.env.tcp_for_joint_position(self.goal_joint_pos)
        self.previous_action = np.zeros(6, dtype=np.float32)
        self.episode_step = 0
        self.previous_distance_m = 0.0
        self.reset()

    def observation(self) -> np.ndarray:
        return build_observation(
            self.config,
            self.env.joint_position(),
            self.env.joint_velocity(),
            self.env.tcp_position(),
            self.goal_position,
            self.previous_action,
        )

    def reset(self) -> np.ndarray:
        self.env.reset(INITIAL_JOINT_POS)
        self.previous_action.fill(0.0)
        self.episode_step = 0
        self.previous_distance_m = float(np.linalg.norm(self.goal_position - self.env.tcp_position()))
        return self.observation()

    def step(self, action: np.ndarray) -> ReachStep:
        bounded_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if bounded_action.shape != (6,):
            raise ValueError(f"action must have shape (6,), got {bounded_action.shape}")

        self.env.apply_action(bounded_action)
        self.env.step()
        self.env.assert_finite_and_within_limits()
        self.episode_step += 1

        distance_m = float(np.linalg.norm(self.goal_position - self.env.tcp_position()))
        reward, terminated, succeeded = compute_reward(
            self.config,
            self.previous_distance_m,
            distance_m,
            bounded_action,
        )
        truncated = self.episode_step >= EPISODE_LENGTH_STEPS and not terminated
        self.previous_distance_m = distance_m
        self.previous_action = bounded_action.copy()
        return ReachStep(
            observation=self.observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            succeeded=succeeded,
            distance_m=distance_m,
        )

    def joint_space_oracle_action(self) -> np.ndarray:
        """Diagnostic controller used only to prove that task success is reachable."""
        error = self.goal_joint_pos - self.env.joint_position()
        # A gain above one compensates the position-drive steady-state error
        # caused by gravity acting through the long scoop.  This controller is
        # not used by PPO; the policy still emits ordinary bounded actions.
        return np.clip(2.0 * error / self.config.action_scale_rad, -1.0, 1.0).astype(np.float32)
