"""Minimal single-environment EC66 reach simulation for Gate 2A."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from .scene_builder import BODY_NAMES, INITIAL_JOINT_POS, build_ec66_model


CONTROL_DT_S = 0.05
PHYSICS_SUBSTEPS = 4
ACTION_SCALE_RAD = 0.04

# The selected TCP is the center of the scoop's cutting edge.  The scoop plate
# spans x=[0.20, 0.63] m in scoop_link coordinates and is centered at z=-0.025
# m.  Applying flange_joint and scoop_mount gives this link6-local offset.
TCP_OFFSET_LINK6 = np.array([-0.025, -0.719, -0.00047661], dtype=np.float32)


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a Newton/Warp quaternion stored as (x, y, z, w)."""
    xyz = quaternion[:3]
    w = float(quaternion[3])
    cross = np.cross(xyz, vector)
    return vector + 2.0 * (w * cross + np.cross(xyz, cross))


class Ec66ReachEnv:
    """One fixed-base EC66 driven by bounded joint-position targets."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        control_dt_s: float = CONTROL_DT_S,
        physics_substeps: int = PHYSICS_SUBSTEPS,
    ) -> None:
        if control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if physics_substeps < 1:
            raise ValueError("physics_substeps must be at least one")

        self.control_dt_s = float(control_dt_s)
        self.physics_substeps = int(physics_substeps)
        self.physics_dt_s = self.control_dt_s / self.physics_substeps

        self.model = build_ec66_model(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=True,
            use_mujoco_contacts=False,
            njmax=64,
            nconmax=16,
        )
        # Newton 1.2.1 still expects a Contacts object when contacts are disabled.
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        self.link6_index = self.model.body_label.index(BODY_NAMES[-1])
        self.joint_lower = self.model.joint_limit_lower.numpy()[:6].astype(np.float32, copy=True)
        self.joint_upper = self.model.joint_limit_upper.numpy()[:6].astype(np.float32, copy=True)
        self.target_joint_pos = INITIAL_JOINT_POS.copy()
        self.kinematic_state = self.model.state()

        self.control.joint_target_vel.zero_()
        self.set_joint_target(self.target_joint_pos)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def joint_position(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[:6].astype(np.float32, copy=True)

    def joint_velocity(self) -> np.ndarray:
        return self.state_0.joint_qd.numpy()[:6].astype(np.float32, copy=True)

    def _tcp_from_state(self, state: newton.State) -> np.ndarray:
        link6_pose = state.body_q.numpy()[self.link6_index]
        position = link6_pose[:3]
        quaternion = link6_pose[3:7]
        return (position + _rotate_xyzw(quaternion, TCP_OFFSET_LINK6)).astype(np.float32)

    def tcp_position(self) -> np.ndarray:
        return self._tcp_from_state(self.state_0)

    def tcp_for_joint_position(self, joint_pos: np.ndarray) -> np.ndarray:
        requested = np.asarray(joint_pos, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"joint_pos must have shape (6,), got {requested.shape}")
        if np.any(requested < self.joint_lower) or np.any(requested > self.joint_upper):
            raise ValueError("joint_pos violates an EC66 joint limit")
        self.kinematic_state.joint_q.assign(
            wp.array(requested, dtype=wp.float32, device=self.model.device)
        )
        self.kinematic_state.joint_qd.zero_()
        newton.eval_fk(
            self.model,
            self.kinematic_state.joint_q,
            self.kinematic_state.joint_qd,
            self.kinematic_state,
        )
        return self._tcp_from_state(self.kinematic_state)

    def reset(self, joint_pos: np.ndarray | None = None) -> None:
        requested = INITIAL_JOINT_POS if joint_pos is None else np.asarray(joint_pos, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"joint_pos must have shape (6,), got {requested.shape}")
        bounded = np.clip(requested, self.joint_lower, self.joint_upper).astype(np.float32)
        q_wp = wp.array(bounded, dtype=wp.float32, device=self.model.device)
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(q_wp)
            state.joint_qd.zero_()
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.control.joint_target_vel.zero_()
        self.set_joint_target(bounded)

    def set_joint_target(self, target: np.ndarray) -> None:
        requested = np.asarray(target, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"target must have shape (6,), got {requested.shape}")
        if not np.isfinite(requested).all():
            raise ValueError("target contains NaN or infinity")
        bounded = np.clip(requested, self.joint_lower, self.joint_upper)
        self.target_joint_pos = bounded.astype(np.float32, copy=True)
        self.control.joint_target_pos.assign(
            wp.array(self.target_joint_pos, dtype=wp.float32, device=self.model.device)
        )

    def apply_action(self, action: np.ndarray) -> None:
        """Apply one bounded Reach-v0 action relative to the current posture."""
        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (6,):
            raise ValueError(f"action must have shape (6,), got {requested.shape}")
        bounded = np.clip(requested, -1.0, 1.0)
        self.set_joint_target(self.joint_position() + bounded * ACTION_SCALE_RAD)

    def step(self) -> None:
        for _ in range(self.physics_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.physics_dt_s,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def assert_finite_and_within_limits(self, *, limit_tolerance: float = 1.0e-4) -> None:
        q = self.joint_position()
        qd = self.joint_velocity()
        tcp = self.tcp_position()
        if not np.isfinite(q).all() or not np.isfinite(qd).all() or not np.isfinite(tcp).all():
            raise AssertionError(f"Non-finite EC66 state: q={q}, qd={qd}, tcp={tcp}")
        if np.any(q < self.joint_lower - limit_tolerance) or np.any(q > self.joint_upper + limit_tolerance):
            raise AssertionError(f"EC66 joint limit violation: q={q}")
