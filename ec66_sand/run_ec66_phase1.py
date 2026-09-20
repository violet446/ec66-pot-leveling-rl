"""Newton phase 1: simplified EC66 tracking the Blender reference trajectory."""

from __future__ import annotations

import argparse
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.controllers import ControllerJointImpedance

from ec66_scene import add_bin, add_ec66, add_pedestal
from ec66_trajectory import BASE_POSITION, build_reference_trajectory


@wp.kernel
def clamp_joint_effort(
    effort: wp.array(dtype=wp.float32),
    effort_limit: wp.array(dtype=wp.float32),
):
    dof = wp.tid()
    limit = effort_limit[dof]
    effort[dof] = wp.clamp(effort[dof], -limit, limit)


class Example:
    include_bin = False
    enable_contacts = False

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.reference = build_reference_trajectory()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.7
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 1.0e3

        initial_q = self.reference.sample(0.0, loop=False).astype(np.float32)
        add_ec66(builder, initial_q, args.kp, args.kd, args.controller)

        # Static pedestal: visual context only in phase 1.  The robot base is
        # fixed by its base joint and does not rely on pedestal contact.
        add_pedestal(builder)
        if self.include_bin:
            add_bin(builder, segments=args.bin_segments)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.impedance_controller = None
        if args.controller == "impedance":
            self.impedance_controller = ControllerJointImpedance(
                self.model,
                stiffness=float(args.kp),
                damping=float(args.kd),
                use_gravity_compensation=True,
                use_coriolis_compensation=True,
                use_inertia_decoupling=True,
            )
            self.impedance_input = self.impedance_controller.input()
            self.impedance_output = self.impedance_controller.output()
            self.impedance_output.joint_f = self.control.joint_f[self.impedance_controller.qd_start]
            self.impedance_input.joint_q = self.state_0.joint_q
            self.impedance_input.joint_qd = self.state_0.joint_qd

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=not self.enable_contacts,
            use_mujoco_contacts=self.enable_contacts,
            njmax=300,
        )
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0) if self.enable_contacts else None
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.8, -4.5, 2.7), pitch=-12.0, yaw=145.0)

        self.max_tracking_error = np.zeros(6, dtype=np.float64)
        self.max_command_effort = np.zeros(6, dtype=np.float64)
        self._set_target(0.0)

    def _set_target(self, time_seconds: float) -> None:
        target = self.reference.sample(time_seconds, loop=self.args.loop).astype(np.float32)
        if self.impedance_controller is None:
            self.control.joint_target_q.assign(wp.array(target, dtype=wp.float32, device=self.device))
        else:
            self.impedance_input.joint_q_des.assign(target)

    def simulate(self):
        if self.impedance_controller is not None:
            self.impedance_controller.step(
                inputs=self.impedance_input,
                outputs=self.impedance_output,
                dt=self.sim_dt,
            )
            wp.launch(
                clamp_joint_effort,
                dim=6,
                inputs=[self.control.joint_f, self.model.joint_effort_limit],
                device=self.device,
            )
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if self.enable_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        # Drive toward the state at the end of this integration interval. Using
        # the interval-start target creates an avoidable one-frame phase lag.
        target_time = self.sim_time + self.frame_dt
        self._set_target(target_time)
        self.simulate()
        actual = self.state_0.joint_q.numpy()[:6]
        target = self.reference.sample(target_time, loop=self.args.loop)
        self.max_tracking_error = np.maximum(self.max_tracking_error, np.abs(actual - target))
        if self.impedance_controller is not None:
            self.max_command_effort = np.maximum(self.max_command_effort, np.abs(self.control.joint_f.numpy()[:6]))
        self.sim_time = target_time

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.enable_contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()[:6]
        if not np.isfinite(joint_q).all():
            raise AssertionError("EC66 joint state contains NaN or infinity")
        print("EC66_PHASE1_MAX_TRACKING_ERROR_RAD=" + np.array2string(self.max_tracking_error, precision=6))
        print("EC66_PHASE1_MAX_TRACKING_ERROR_DEG=" + np.array2string(np.degrees(self.max_tracking_error), precision=3))
        if self.impedance_controller is not None:
            print("EC66_MAX_COMMAND_EFFORT_NM=" + np.array2string(self.max_command_effort, precision=3))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--fps", type=float, default=60.0, help="Newton display/control frequency")
        parser.add_argument("--substeps", type=int, default=8, help="MuJoCo substeps per displayed frame")
        parser.add_argument("--kp", type=float, default=1200.0, help="provisional joint position gain")
        parser.add_argument("--kd", type=float, default=80.0, help="provisional joint velocity gain")
        parser.add_argument(
            "--controller",
            choices=("drive", "impedance"),
            default="drive",
            help="MuJoCo position drive or gravity-compensated effort controller",
        )
        parser.add_argument("--bin-segments", type=int, default=48, help="number of overlapping wall proxies")
        parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="loop the 9 second reference")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, arguments = newton.examples.init(parser)
    newton.examples.run(Example(viewer, arguments), arguments)
