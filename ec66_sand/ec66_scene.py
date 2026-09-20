"""Shared EC66 and bin construction helpers for all simulation phases."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode

from ec66_trajectory import BASE_POSITION


PROJECT_DIR = Path(__file__).resolve().parent
ROBOT_URDF = PROJECT_DIR / "assets" / "ec66_simplified.urdf"

BIN_INNER_RADIUS = 1.20
BIN_WALL_THICKNESS = 0.025
BIN_BOTTOM_Z = 0.055
BIN_FLOOR_Z = 0.08
BIN_RIM_Z = 0.78
ROBOT_CONTACT_GAP = 0.0025
BIN_CONTACT_GAP = 0.0025


def add_ec66(
    builder: newton.ModelBuilder,
    initial_q: np.ndarray,
    kp: float,
    kd: float,
    controller_mode: str = "drive",
) -> None:
    """Add the simplified fixed-base EC66 and configure its position drives."""
    # Newton otherwise derives a 0.1 m gap for these primitives.  That is useful
    # for broad speculative contact, but far too large for a robot clearing a
    # bin rim: two such shells can react while the meshes are still 0.2 m apart.
    builder.default_shape_cfg.gap = ROBOT_CONTACT_GAP
    builder.add_urdf(
        str(ROBOT_URDF),
        xform=wp.transform(wp.vec3(*BASE_POSITION), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
        force_position_velocity_actuation=True,
    )
    if builder.joint_dof_count != 6:
        raise RuntimeError(f"Expected 6 EC66 DOFs after fixed-joint collapse, got {builder.joint_dof_count}")

    builder.joint_q[:] = initial_q.tolist()
    builder.joint_target_q[:] = initial_q.tolist()
    for dof in range(builder.joint_dof_count):
        builder.joint_target_ke[dof] = float(kp)
        builder.joint_target_kd[dof] = float(kd)
        builder.joint_target_mode[dof] = int(
            JointTargetMode.EFFORT if controller_mode == "impedance" else JointTargetMode.POSITION
        )


def add_pedestal(builder: newton.ModelBuilder) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0, gap=BIN_CONTACT_GAP)
    shape = builder.add_shape_cylinder(
        -1,
        xform=wp.transform(
            wp.vec3(float(BASE_POSITION[0]), float(BASE_POSITION[1]), 0.475),
            wp.quat_identity(),
        ),
        radius=0.075,
        half_height=0.475,
        cfg=cfg,
        color=wp.vec3(0.72, 0.75, 0.77),
        label="pedestal",
    )
    # The fixed robot base overlaps this cosmetic support by construction.
    # Excluding it avoids artificial base contacts; it is not a sand boundary.
    builder.shape_flags[shape] = int(newton.ShapeFlags.VISIBLE)


def add_bin(builder: newton.ModelBuilder, segments: int = 48) -> None:
    """Build a closed, MPM-friendly circular bin from overlapping primitives.

    The wall uses boxes rather than a thin concave triangle mesh. This gives
    robust normals and avoids gaps at segment joins for both rigid and MPM
    collision paths.
    """
    if segments < 12:
        raise ValueError("The circular bin needs at least 12 wall segments")

    steel = wp.vec3(0.34, 0.39, 0.43)
    floor_cfg = newton.ModelBuilder.ShapeConfig(mu=0.65, density=0.0, gap=BIN_CONTACT_GAP)
    wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.55, density=0.0, gap=BIN_CONTACT_GAP)

    bottom_half_height = 0.5 * (BIN_FLOOR_Z - BIN_BOTTOM_Z)
    builder.add_shape_cylinder(
        -1,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, BIN_BOTTOM_Z + bottom_half_height),
            wp.quat_identity(),
        ),
        radius=BIN_INNER_RADIUS + BIN_WALL_THICKNESS,
        half_height=bottom_half_height,
        cfg=floor_cfg,
        color=steel,
        label="bin_bottom",
    )

    center_radius = BIN_INNER_RADIUS + 0.5 * BIN_WALL_THICKNESS
    half_radial = 0.5 * BIN_WALL_THICKNESS
    half_tangent = center_radius * math.tan(math.pi / segments) * 1.03
    half_height = 0.5 * (BIN_RIM_Z - BIN_FLOOR_Z)
    center_z = 0.5 * (BIN_RIM_Z + BIN_FLOOR_Z)
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        position = wp.vec3(center_radius * math.cos(angle), center_radius * math.sin(angle), center_z)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        builder.add_shape_box(
            -1,
            xform=wp.transform(position, rotation),
            hx=half_radial,
            hy=half_tangent,
            hz=half_height,
            cfg=wall_cfg,
            color=steel,
            label=f"bin_wall_{index:02d}",
        )
