"""Newton phase 3: EC66 moving through a low-resolution implicit-MPM sand bed.

The coupling is intentionally one-way in this phase: MuJoCo supplies the robot
pose and velocity to MPM.  MPM reaction impulses are measured, but are not yet
applied back to the robot generalized coordinates.
"""

from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.controllers import ControllerJointImpedance
from newton.solvers import SolverImplicitMPM

from ec66_scene import BIN_FLOOR_Z, BIN_INNER_RADIUS, add_bin, add_ec66, add_pedestal
from ec66_trajectory import build_reference_trajectory
from run_ec66_phase1 import clamp_joint_effort


def dig_z_offset(frame: float, depth: float) -> float:
    """Smoothly lower only the digging portion of the Blender tool path."""
    def smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    if frame <= 29.0 or frame >= 118.0:
        return 0.0
    if frame < 54.0:
        return -depth * smoothstep((frame - 29.0) / 25.0)
    if frame <= 92.0:
        return -depth
    return -depth * (1.0 - smoothstep((frame - 92.0) / 26.0))


@wp.kernel
def accumulate_sand_reaction(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    collider_body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Accumulate the reaction wrench measured at each rigid-body COM."""
    i = wp.tid()
    collider_id = collider_ids[i]
    if collider_id < 0 or collider_id >= collider_body_ids.shape[0]:
        return
    body_id = collider_body_ids[collider_id]
    if body_id < 0:
        return
    force = collider_impulses[i] / dt
    center = wp.transform_point(body_q[body_id], body_com[body_id])
    torque = wp.cross(collider_impulse_pos[i] - center, force)
    wp.atomic_add(body_wrench, body_id, wp.spatial_vector(force, torque))


def add_circular_sand_bed(
    builder: newton.ModelBuilder,
    voxel_size: float,
    particles_per_cell: float,
    density: float,
    friction: float,
    young_modulus: float,
    poisson_ratio: float,
    seed: int,
) -> tuple[int, float, float]:
    """Create a deterministic particle lattice clipped to the Blender sand surface."""
    if voxel_size <= 0.0 or particles_per_cell <= 0.0:
        raise ValueError("voxel size and particles per cell must both be positive")
    if density <= 0.0:
        raise ValueError("sand density must be positive")
    spacing = voxel_size / particles_per_cell
    radius_limit = BIN_INNER_RADIUS - 1.5 * spacing
    z_lo = BIN_FLOOR_Z + 0.65 * spacing
    rng = np.random.default_rng(seed)

    coordinates: list[wp.vec3] = []
    x_values = np.arange(-radius_limit, radius_limit + 0.5 * spacing, spacing)
    y_values = np.arange(-radius_limit, radius_limit + 0.5 * spacing, spacing)
    for x in x_values:
        for y in y_values:
            if x * x + y * y > radius_limit * radius_limit:
                continue
            surface_z = 0.53 + 0.18 * math.exp(-2.8 * (x * x + y * y)) + 0.012 * math.sin(9.0 * x) * math.cos(7.0 * y)
            z_values = np.arange(z_lo, surface_z - 0.35 * spacing, spacing)
            for z in z_values:
                # Small deterministic jitter breaks lattice symmetry while
                # keeping every initial particle clear of the boundaries.
                jitter = rng.uniform(-0.12, 0.12, size=3) * spacing
                jitter[2] *= 0.5
                coordinates.append(wp.vec3(float(x + jitter[0]), float(y + jitter[1]), float(z + jitter[2])))

    count = len(coordinates)
    cell_volume = spacing**3
    particle_mass = density * cell_volume
    particle_radius = 0.5 * spacing
    zeros = [wp.vec3(0.0)] * count
    builder.add_particles(
        pos=coordinates,
        vel=zeros,
        mass=[particle_mass] * count,
        radius=[particle_radius] * count,
        custom_attributes={
            "mpm:young_modulus": [young_modulus] * count,
            "mpm:poisson_ratio": [poisson_ratio] * count,
            "mpm:friction": [friction] * count,
            "mpm:tensile_yield_ratio": [0.0] * count,
            "mpm:yield_pressure": [1.0e12] * count,
        },
    )
    return count, particle_mass, particle_radius


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.rigid_substeps = int(args.rigid_substeps)
        self.rigid_dt = self.frame_dt / self.rigid_substeps
        self.mpm_substeps = int(args.mpm_substeps)
        self.mpm_dt = self.frame_dt / self.mpm_substeps
        self.sim_time = 0.0
        self.reference = build_reference_trajectory(
            tool_z_offset=lambda frame: dig_z_offset(frame, float(args.dig_depth_offset))
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.7
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 1.0e3

        initial_q = self.reference.sample(0.0, loop=False).astype(np.float32)
        add_ec66(builder, initial_q, args.kp, args.kd, controller_mode="impedance")
        add_pedestal(builder)
        add_bin(builder, segments=args.bin_segments)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
        self.particle_count, self.particle_mass, self.particle_radius = add_circular_sand_bed(
            builder,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            density=args.density,
            friction=args.friction,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            seed=args.seed,
        )

        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.controller = ControllerJointImpedance(
            self.model,
            stiffness=float(args.kp),
            damping=float(args.kd),
            use_gravity_compensation=True,
            use_coriolis_compensation=True,
            use_inertia_decoupling=True,
        )
        self.controller_input = self.controller.input()
        self.controller_output = self.controller.output()
        self.controller_output.joint_f = self.control.joint_f[self.controller.qd_start]
        self.controller_input.joint_q = self.state_0.joint_q
        self.controller_input.joint_qd = self.state_0.joint_qd

        self.rigid_solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=False,
            use_mujoco_contacts=True,
            njmax=300,
        )
        self.rigid_contacts = newton.Contacts(self.rigid_solver.get_max_contact_count(), 0)

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = float(args.voxel_size)
        mpm_config.grid_type = args.grid_type
        mpm_config.grid_padding = int(args.grid_padding) if args.grid_type == "fixed" else 0
        mpm_config.max_active_cell_count = int(args.max_active_cells) if args.grid_type == "fixed" else -1
        mpm_config.strain_basis = "P0"
        mpm_config.transfer_scheme = "apic"
        mpm_config.integration_scheme = "pic"
        mpm_config.max_iterations = int(args.mpm_iterations)
        mpm_config.tolerance = float(args.mpm_tolerance)
        mpm_config.critical_fraction = 0.0
        mpm_config.air_drag = 0.1
        mpm_config.collider_velocity_mode = "backward"
        self.mpm_solver = SolverImplicitMPM(self.model, config=mpm_config)
        # Zero effective collider mass makes all robot links kinematic from the
        # MPM perspective. Their current state still supplies motion/velocity.
        self.mpm_solver.setup_collider(body_mass=wp.zeros_like(self.model.body_mass), body_q=self.state_0.body_q)

        self.collider_body_ids = self.mpm_solver.collider_body_index
        self.sand_body_wrench = wp.zeros_like(self.state_0.body_f)
        self.max_sand_force = 0.0
        self.max_sand_torque = 0.0
        self.max_tracking_error = np.zeros(6, dtype=np.float64)
        self.max_rigid_contacts = 0

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.58, 0.36, 0.14),
            dtype=wp.vec3,
            device=self.device,
        )
        self.show_impulses = False
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.8, -4.5, 2.7), pitch=-12.0, yaw=145.0)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.robot_graph = None
        self.sand_graph = None
        self._capture_graphs()

        total_mass = self.particle_count * self.particle_mass
        print(f"EC66_PHASE3_PARTICLE_COUNT={self.particle_count}")
        print(f"EC66_PHASE3_INITIAL_SAND_MASS_KG={total_mass:.3f}")
        print(f"EC66_PHASE3_EXTRA_DIG_DEPTH_M={args.dig_depth_offset:.3f}")
        print("EC66_PHASE3_COUPLING=one-way robot-to-MPM; reaction measured but not applied")

    def _capture_graphs(self) -> None:
        if not self.device.is_cuda:
            return
        with wp.ScopedCapture() as capture:
            self._simulate_robot()
        self.robot_graph = capture.graph
        if self.mpm_solver.grid_type == "fixed":
            if self.mpm_substeps % 2:
                warnings.warn("Use an even --mpm-substeps value for fixed-grid CUDA graph capture", stacklevel=2)
            else:
                with wp.ScopedCapture() as capture:
                    self._simulate_sand()
                self.sand_graph = capture.graph

    def _set_target(self, time_seconds: float) -> None:
        target = self.reference.sample(time_seconds, loop=self.args.loop).astype(np.float32)
        self.controller_input.joint_q_des.assign(target)

    def _simulate_robot(self) -> None:
        self.controller.step(inputs=self.controller_input, outputs=self.controller_output, dt=self.rigid_dt)
        wp.launch(
            clamp_joint_effort,
            dim=6,
            inputs=[self.control.joint_f, self.model.joint_effort_limit],
            device=self.device,
        )
        for _ in range(self.rigid_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.rigid_solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.rigid_contacts,
                self.rigid_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.rigid_solver.update_contacts(self.rigid_contacts, self.state_0)

    def _simulate_sand(self) -> None:
        for _ in range(self.mpm_substeps):
            self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.mpm_dt)
            self.mpm_solver.project_outside(self.state_0, self.state_0, self.mpm_dt)

    def _measure_reaction(self) -> tuple[wp.array, wp.array, wp.array]:
        impulses, positions, collider_ids = self.mpm_solver.collect_collider_impulses(self.state_0)
        self.sand_body_wrench.zero_()
        wp.launch(
            accumulate_sand_reaction,
            dim=collider_ids.shape[0],
            inputs=[
                self.frame_dt,
                collider_ids,
                impulses,
                positions,
                self.collider_body_ids,
                self.state_0.body_q,
                self.model.body_com,
                self.sand_body_wrench,
            ],
            device=self.device,
        )
        wrench = self.sand_body_wrench.numpy()
        if wrench.size:
            self.max_sand_force = max(self.max_sand_force, float(np.linalg.norm(wrench[:, :3], axis=1).max()))
            self.max_sand_torque = max(self.max_sand_torque, float(np.linalg.norm(wrench[:, 3:], axis=1).max()))
        return impulses, positions, collider_ids

    def step(self) -> None:
        target_time = self.sim_time + self.frame_dt
        self._set_target(target_time)
        if self.robot_graph is not None:
            wp.capture_launch(self.robot_graph)
        else:
            self._simulate_robot()
        if self.sand_graph is not None:
            wp.capture_launch(self.sand_graph)
        else:
            self._simulate_sand()

        self._measure_reaction()
        actual = self.state_0.joint_q.numpy()[:6]
        target = self.reference.sample(target_time, loop=self.args.loop)
        self.max_tracking_error = np.maximum(self.max_tracking_error, np.abs(actual - target))
        count = int(self.rigid_contacts.rigid_contact_count.numpy()[0])
        self.max_rigid_contacts = max(self.max_rigid_contacts, count)
        self.sim_time = target_time

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.rigid_contacts, self.state_0)
        self.viewer.log_points(
            "/sand",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        if self.show_impulses:
            impulses, positions, _ = self.mpm_solver.collect_collider_impulses(self.state_0)
            scale = 0.02
            self.viewer.log_lines(
                "/sand_reaction_impulses",
                starts=positions,
                ends=positions + scale * impulses,
                colors=wp.full(positions.shape[0], value=wp.vec3(1.0, 0.1, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/sand_reaction_impulses", None, None, None)
        self.viewer.end_frame()

    def render_ui(self, imgui) -> None:
        _changed, self.show_impulses = imgui.checkbox("Show sand impulses", self.show_impulses)

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        if not np.isfinite(particle_q).all() or not np.isfinite(particle_qd).all():
            raise AssertionError("MPM particle state contains NaN or infinity")
        min_z = float(particle_q[:, 2].min())
        max_radius = float(np.linalg.norm(particle_q[:, :2], axis=1).max())
        max_speed = float(np.linalg.norm(particle_qd, axis=1).max())
        print(f"EC66_PHASE3_PARTICLE_Z_RANGE_M=({min_z:.4f}, {float(particle_q[:, 2].max()):.4f})")
        print(f"EC66_PHASE3_MAX_PARTICLE_RADIUS_M={max_radius:.4f}")
        print(f"EC66_PHASE3_MAX_PARTICLE_SPEED_M_S={max_speed:.4f}")
        print(f"EC66_PHASE3_MAX_MEASURED_BODY_FORCE_N={self.max_sand_force:.3f}")
        print(f"EC66_PHASE3_MAX_MEASURED_BODY_TORQUE_NM={self.max_sand_torque:.3f}")
        print("EC66_PHASE3_MAX_TRACKING_ERROR_DEG=" + np.array2string(np.degrees(self.max_tracking_error), precision=3))
        print(f"EC66_PHASE3_MAX_RIGID_CONTACT_COUNT={self.max_rigid_contacts}")
        if min_z < BIN_FLOOR_Z - 1.1 * self.args.voxel_size:
            raise AssertionError(f"Particles penetrated below the bin floor: min z={min_z:.4f} m")
        if max_radius > BIN_INNER_RADIUS + 1.5 * self.args.voxel_size:
            raise AssertionError(f"Particles escaped through the bin wall: max radius={max_radius:.4f} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--rigid-substeps", type=int, default=8)
        parser.add_argument("--mpm-substeps", type=int, default=2)
        parser.add_argument("--kp", type=float, default=1200.0)
        parser.add_argument("--kd", type=float, default=80.0)
        parser.add_argument("--bin-segments", type=int, default=48)
        parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--voxel-size", type=float, default=0.08)
        parser.add_argument("--particles-per-cell", type=float, default=2.0)
        parser.add_argument("--density", type=float, default=1600.0)
        parser.add_argument("--friction", type=float, default=0.68)
        parser.add_argument("--young-modulus", type=float, default=5.0e7)
        parser.add_argument("--poisson-ratio", type=float, default=0.2)
        parser.add_argument("--grid-type", choices=("fixed", "sparse", "dense"), default="fixed")
        parser.add_argument("--grid-padding", type=int, default=10)
        parser.add_argument("--max-active-cells", type=int, default=1 << 18)
        parser.add_argument("--mpm-iterations", type=int, default=50)
        parser.add_argument("--mpm-tolerance", type=float, default=1.0e-4)
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument(
            "--dig-depth-offset",
            type=float,
            default=0.12,
            help="extra downward tool offset during frames 29-118, in metres",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, arguments = newton.examples.init(parser)
    newton.examples.run(Example(viewer, arguments), arguments)
