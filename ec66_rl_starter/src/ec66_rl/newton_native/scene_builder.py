"""Build the fixed-base EC66 model for the Newton 1.2.1 RL environment.

This module intentionally uses only APIs available in the isolated
``env_isaaclab`` environment.  In particular, it does not import
``newton.controllers`` from the separate Newton 1.6 sand project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EC66_URDF = PROJECT_ROOT.parent / "ec66_sand" / "assets" / "ec66_simplified.urdf"

JOINT_NAMES = tuple(f"ec66_simplified/joint{index}" for index in range(1, 7))
BODY_NAMES = tuple(f"ec66_simplified/link{index}" for index in range(1, 7))

# A collision-free pose already used by the validated EC66 reference motion.
INITIAL_JOINT_POS = np.array(
    [-0.001132283, -0.979132976, 2.058742307, -4.221201981, 1.571928620, 1.570796329],
    dtype=np.float32,
)
URDF_VELOCITY_LIMITS = np.array([2.6199, 2.6199, 3.3161, 4.5379, 4.5379, 4.5379], dtype=np.float32)

DEFAULT_KP = 1200.0
DEFAULT_KD = 80.0


def _build_ec66_template(*, kp: float, kd: float, urdf_path: Path = EC66_URDF) -> newton.ModelBuilder:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"EC66 URDF not found: {urdf_path}")
    if kp <= 0.0 or kd < 0.0:
        raise ValueError(f"Invalid drive gains: kp={kp}, kd={kd}")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_urdf(
        str(urdf_path),
        xform=wp.transform_identity(),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
        force_position_velocity_actuation=True,
    )

    if builder.joint_dof_count != 6 or builder.joint_coord_count != 6:
        raise RuntimeError(
            f"Expected 6 EC66 coordinates/DOFs, got coordinates={builder.joint_coord_count}, "
            f"DOFs={builder.joint_dof_count}"
        )
    if tuple(builder.joint_label) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint order: {builder.joint_label}")
    if tuple(builder.body_label) != BODY_NAMES:
        raise RuntimeError(f"Unexpected body order: {builder.body_label}")

    builder.joint_q[:] = INITIAL_JOINT_POS.tolist()
    builder.joint_target_pos[:] = INITIAL_JOINT_POS.tolist()
    builder.joint_velocity_limit[:] = URDF_VELOCITY_LIMITS.tolist()
    for dof in range(6):
        builder.joint_target_ke[dof] = float(kp)
        builder.joint_target_kd[dof] = float(kd)
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)

    return builder


def build_ec66_model(
    device: str = "cuda:0",
    *,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
    num_envs: int = 1,
) -> newton.Model:
    """Load and finalize one or more independent fixed-base EC66 worlds."""
    if num_envs < 1:
        raise ValueError("num_envs must be at least one")
    template = _build_ec66_template(kp=kp, kd=kd)
    if num_envs == 1:
        builder = template
    else:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, world_count=num_envs)

    model = builder.finalize(device=device)
    lower = model.joint_limit_lower.numpy()[:6]
    upper = model.joint_limit_upper.numpy()[:6]
    if np.any(INITIAL_JOINT_POS < lower) or np.any(INITIAL_JOINT_POS > upper):
        raise RuntimeError("The Gate 2A initial pose violates an EC66 joint limit")
    return model
