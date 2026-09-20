"""RSL-RL VecEnv adapter for fixed or random-goal EC66 Reach-v0."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .batched_env import BatchedEc66Env
from .reach_env import ACTION_SCALE_RAD
from .reach_task import EPISODE_LENGTH_STEPS, FIXED_GOAL_JOINT_POS


SAFE_GOAL_DELTA_RAD = torch.tensor([0.30, 0.25, 0.25, 0.20, 0.20, 0.20])
SAFE_START_DELTA_RAD = torch.tensor([0.20, 0.18, 0.18, 0.12, 0.12, 0.12])
MIN_INITIAL_GOAL_DISTANCE_M = 0.10
MAX_INITIAL_GOAL_DISTANCE_M = 0.45
GOAL_DISTANCE_BANDS_M = {
    "near": (0.10, 0.20),
    "medium": (0.20, 0.32),
    "far": (0.32, 0.45),
}


class Ec66RslRlEnv(VecEnv):
    num_actions = 6
    max_episode_length = EPISODE_LENGTH_STEPS

    def __init__(
        self,
        num_envs: int = 16,
        *,
        device: str = "cuda:0",
        random_goals: bool = True,
        random_starts: bool = False,
        goal_distance_band: str | None = None,
        seed: int = 42,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.random_goals = bool(random_goals)
        self.random_starts = bool(random_starts)
        if goal_distance_band is not None and goal_distance_band not in GOAL_DISTANCE_BANDS_M:
            raise ValueError(
                f"goal_distance_band must be one of {tuple(GOAL_DISTANCE_BANDS_M)}, got {goal_distance_band!r}"
            )
        self.goal_distance_band = goal_distance_band
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.cfg = {
            "task_name": "EC66-Reach-Random-v0" if self.random_goals else "EC66-Reach-Fixed-v0",
            "num_envs": self.num_envs,
            "tcp": "scoop_cutting_edge_center",
            "control_dt_s": 0.05,
        }
        self.physics = BatchedEc66Env(self.num_envs, device=device)
        self.goal_joint_pos = torch.as_tensor(FIXED_GOAL_JOINT_POS, device=self.device).repeat(self.num_envs, 1)
        self.goal_joint_pos.clamp_(self.physics.joint_lower, self.physics.joint_upper)
        self.goal_position = self.physics.tcp_for_joint_positions(self.goal_joint_pos)

        self.previous_action = torch.zeros((self.num_envs, 6), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_distance_min = torch.full(
            (self.num_envs,), MIN_INITIAL_GOAL_DISTANCE_M, device=self.device
        )
        self.goal_distance_max = torch.full(
            (self.num_envs,), MAX_INITIAL_GOAL_DISTANCE_M, device=self.device
        )
        self.reset()

    def _observations_tensor(self) -> torch.Tensor:
        q = self.physics.joint_position
        qd = self.physics.joint_velocity
        q_normalized = 2.0 * (q - self.physics.joint_lower) / (
            self.physics.joint_upper - self.physics.joint_lower
        ) - 1.0
        qd_normalized = (qd / 2.0).clamp(-1.0, 1.0)
        tcp_to_goal = self.goal_position - self.physics.tcp_position
        distance = torch.linalg.vector_norm(tcp_to_goal, dim=-1, keepdim=True)
        return torch.cat(
            (q_normalized, qd_normalized, tcp_to_goal, self.previous_action, distance), dim=-1
        )

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._observations_tensor()}, batch_size=[self.num_envs])

    def _sample_initial_joint_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        delta_limit = SAFE_START_DELTA_RAD.to(self.device)
        random_delta = (
            2.0 * torch.rand((env_ids.numel(), 6), generator=self.rng, device=self.device) - 1.0
        ) * delta_limit
        return (self.physics.initial_joint_pos[env_ids] + random_delta).clamp(
            self.physics.joint_lower[env_ids], self.physics.joint_upper[env_ids]
        )

    def _assign_goal_distance_bounds(self, env_ids: torch.Tensor) -> None:
        if self.goal_distance_band is not None:
            minimum, maximum = GOAL_DISTANCE_BANDS_M[self.goal_distance_band]
            self.goal_distance_min[env_ids] = minimum
            self.goal_distance_max[env_ids] = maximum
            return
        band_names = tuple(GOAL_DISTANCE_BANDS_M)
        band_ids = torch.randint(
            len(band_names),
            (env_ids.numel(),),
            generator=self.rng,
            device=self.device,
        )
        for band_id, band_name in enumerate(band_names):
            selected = env_ids[band_ids == band_id]
            if selected.numel() > 0:
                minimum, maximum = GOAL_DISTANCE_BANDS_M[band_name]
                self.goal_distance_min[selected] = minimum
                self.goal_distance_max[selected] = maximum

    def _sample_reachable_goals(self, env_ids: torch.Tensor) -> None:
        """Sample safe joint poses and use FK to produce guaranteed-reachable goals."""
        self._assign_goal_distance_bounds(env_ids)
        pending = env_ids
        delta_limit = SAFE_GOAL_DELTA_RAD.to(self.device)
        initial_tcp = self.physics.tcp_position
        max_sampling_attempts = 256
        for _ in range(max_sampling_attempts):
            if pending.numel() == 0:
                break
            random_delta = (
                2.0 * torch.rand((pending.numel(), 6), generator=self.rng, device=self.device) - 1.0
            ) * delta_limit
            candidates = (self.physics.joint_position[pending] + random_delta).clamp(
                self.physics.joint_lower[pending], self.physics.joint_upper[pending]
            )
            trial_q = self.goal_joint_pos.clone()
            trial_q[pending] = candidates
            trial_tcp = self.physics.tcp_for_joint_positions(trial_q)
            distances = torch.linalg.vector_norm(trial_tcp[pending] - initial_tcp[pending], dim=-1)
            accepted = (distances >= self.goal_distance_min[pending]) & (
                distances <= self.goal_distance_max[pending]
            )
            accepted_ids = pending[accepted]
            if accepted_ids.numel() > 0:
                self.goal_joint_pos[accepted_ids] = candidates[accepted]
                self.goal_position[accepted_ids] = trial_tcp[accepted_ids]
            pending = pending[~accepted]
        if pending.numel() > 0:
            pending_bands = [
                (
                    float(self.goal_distance_min[index].item()),
                    float(self.goal_distance_max[index].item()),
                )
                for index in pending
            ]
            raise RuntimeError(
                f"Could not sample {pending.numel()} safe reachable goals after "
                f"{max_sampling_attempts} attempts; requested bands={pending_bands}"
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> TensorDict:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        start_positions = (
            self._sample_initial_joint_positions(env_ids) if self.random_starts else None
        )
        self.physics.reset(env_ids, joint_positions=start_positions)
        if self.random_goals:
            self._sample_reachable_goals(env_ids)
        else:
            fixed_q = torch.as_tensor(FIXED_GOAL_JOINT_POS, device=self.device).repeat(env_ids.numel(), 1)
            self.goal_joint_pos[env_ids] = fixed_q.clamp(
                self.physics.joint_lower[env_ids], self.physics.joint_upper[env_ids]
            )
            self.goal_position.copy_(self.physics.tcp_for_joint_positions(self.goal_joint_pos))
        self.previous_action[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.previous_distance[env_ids] = torch.linalg.vector_norm(
            self.goal_position[env_ids] - self.physics.tcp_position[env_ids], dim=-1
        )
        return self.get_observations()

    def step(
        self, actions: torch.Tensor, *, auto_reset: bool = True
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        bounded = actions.to(self.device).clamp(-1.0, 1.0)
        self.physics.apply_actions(bounded)
        self.physics.step()
        self.episode_length_buf += 1

        distance = torch.linalg.vector_norm(self.goal_position - self.physics.tcp_position, dim=-1)
        progress = self.previous_distance - distance
        rewards = 2.0 * progress - 0.2 * distance - 0.01 * torch.mean(bounded.square(), dim=-1)
        successes = distance <= 0.03
        rewards = rewards + 10.0 * successes.float()
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = successes | time_outs

        self.episode_return += rewards
        # Keep the persistent buffer identity.  RSL-RL may collect rollouts in
        # inference mode; rebinding here would turn the buffer into an
        # inference tensor that cannot later be updated during reset.
        self.previous_distance.copy_(distance)
        self.previous_action.copy_(bounded)

        extras: dict = {
            "time_outs": time_outs,
            "successes": successes,
            "distance": distance.detach(),
        }
        if torch.any(dones):
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            extras["log"] = {
                "/episode_return": self.episode_return[done_ids].mean(),
                "/success_rate": successes[done_ids].float().mean(),
                "/final_distance_m": distance[done_ids].mean(),
            }
            if auto_reset:
                self.reset(done_ids)

        return self.get_observations(), rewards, dones.to(dtype=torch.long), extras
