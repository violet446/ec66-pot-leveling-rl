"""GPU-batched EC66 physics used by the RSL-RL adapter."""

from __future__ import annotations

import torch
import warp as wp

import newton

from .reach_env import ACTION_SCALE_RAD, CONTROL_DT_S, PHYSICS_SUBSTEPS, TCP_OFFSET_LINK6
from .scene_builder import INITIAL_JOINT_POS, build_ec66_model


class BatchedEc66Env:
    """Replicate the fixed-base robot into independent Newton worlds."""

    def __init__(self, num_envs: int, *, device: str = "cuda:0") -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.model = build_ec66_model(device=device, num_envs=self.num_envs)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.kinematic_state = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            separate_worlds=self.num_envs > 1,
            disable_contacts=True,
            use_mujoco_contacts=False,
            njmax=max(64, self.num_envs * 8),
            nconmax=max(16, self.num_envs * 2),
        )
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        self.physics_dt_s = CONTROL_DT_S / PHYSICS_SUBSTEPS

        self.initial_joint_pos = torch.as_tensor(INITIAL_JOINT_POS, device=self.device).repeat(self.num_envs, 1)
        self.joint_lower = wp.to_torch(self.model.joint_limit_lower).view(self.num_envs, 6)
        self.joint_upper = wp.to_torch(self.model.joint_limit_upper).view(self.num_envs, 6)
        self.control.joint_target_vel.zero_()
        self.reset()

    def _q(self, state: newton.State | None = None) -> torch.Tensor:
        selected = self.state_0 if state is None else state
        return wp.to_torch(selected.joint_q).view(self.num_envs, 6)

    def _qd(self, state: newton.State | None = None) -> torch.Tensor:
        selected = self.state_0 if state is None else state
        return wp.to_torch(selected.joint_qd).view(self.num_envs, 6)

    @property
    def joint_position(self) -> torch.Tensor:
        return self._q()

    @property
    def joint_velocity(self) -> torch.Tensor:
        return self._qd()

    def _tcp(self, state: newton.State) -> torch.Tensor:
        poses = wp.to_torch(state.body_q).view(self.num_envs, 6, 7)[:, 5]
        position = poses[:, :3]
        xyz = poses[:, 3:6]
        w = poses[:, 6:7]
        offset = torch.as_tensor(TCP_OFFSET_LINK6, device=self.device).expand(self.num_envs, -1)
        cross = torch.linalg.cross(xyz, offset, dim=-1)
        rotated = offset + 2.0 * (w * cross + torch.linalg.cross(xyz, cross, dim=-1))
        return position + rotated

    @property
    def tcp_position(self) -> torch.Tensor:
        return self._tcp(self.state_0)

    def tcp_for_joint_positions(self, joint_pos: torch.Tensor) -> torch.Tensor:
        requested = joint_pos.to(device=self.device, dtype=torch.float32)
        if requested.shape != (self.num_envs, 6):
            raise ValueError(f"joint_pos must have shape ({self.num_envs}, 6), got {tuple(requested.shape)}")
        self._q(self.kinematic_state).copy_(requested)
        self._qd(self.kinematic_state).zero_()
        newton.eval_fk(
            self.model,
            self.kinematic_state.joint_q,
            self.kinematic_state.joint_qd,
            self.kinematic_state,
        )
        return self._tcp(self.kinematic_state).clone()

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        joint_positions: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if joint_positions is None:
            reset_positions = self.initial_joint_pos[env_ids]
        else:
            reset_positions = joint_positions.to(device=self.device, dtype=torch.float32)
            expected_shape = (env_ids.numel(), 6)
            if reset_positions.shape != expected_shape:
                raise ValueError(
                    f"joint_positions must have shape {expected_shape}, got {tuple(reset_positions.shape)}"
                )
            reset_positions = reset_positions.clamp(
                self.joint_lower[env_ids], self.joint_upper[env_ids]
            )
        for state in (self.state_0, self.state_1):
            self._q(state)[env_ids] = reset_positions
            self._qd(state)[env_ids] = 0.0
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        targets = wp.to_torch(self.control.joint_target_pos).view(self.num_envs, 6)
        targets[env_ids] = reset_positions

    def apply_actions(self, actions: torch.Tensor) -> None:
        bounded = actions.to(device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        targets = (self.joint_position + bounded * ACTION_SCALE_RAD).clamp(self.joint_lower, self.joint_upper)
        wp.to_torch(self.control.joint_target_pos).view(self.num_envs, 6).copy_(targets)

    def step(self) -> None:
        for _ in range(PHYSICS_SUBSTEPS):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.physics_dt_s,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
