"""Open a Newton GL viewer for the fixed-goal EC66 Reach-v0 task."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

from .reach_task import FixedGoalReachTask


class Example:
    def __init__(self, viewer, args) -> None:
        self.viewer = viewer
        self.task = FixedGoalReachTask(device=args.device or "cuda:0")
        self.model = self.task.env.model
        self.state_0 = self.task.env.state_0
        self.state_1 = self.task.env.state_1
        self.control = self.task.env.control
        self.contacts = self.task.env.contacts
        self.sim_time = 0.0
        self.pause_after_success = 0
        self.tcp_radius = wp.array([0.025], dtype=wp.float32, device=self.model.device)
        self.goal_radius = wp.array([0.04], dtype=wp.float32, device=self.model.device)
        self.tcp_color = wp.array([(1.0, 0.2, 0.2)], dtype=wp.vec3, device=self.model.device)
        self.goal_color = wp.array([(0.2, 1.0, 0.2)], dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.8, -2.5, 1.5), pitch=-18.0, yaw=145.0)

    def step(self) -> None:
        if self.pause_after_success > 0:
            self.pause_after_success -= 1
            if self.pause_after_success == 0:
                self.task.reset()
        else:
            result = self.task.step(self.task.joint_space_oracle_action())
            if result.succeeded or result.truncated:
                self.pause_after_success = 30
        self.state_0 = self.task.env.state_0
        self.state_1 = self.task.env.state_1
        self.sim_time += self.task.env.control_dt_s

    def render(self) -> None:
        tcp = wp.array([self.task.env.tcp_position()], dtype=wp.vec3, device=self.model.device)
        goal = wp.array([self.task.goal_position], dtype=wp.vec3, device=self.model.device)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.task.env.state_0)
        self.viewer.log_points("/reach/tcp", tcp, radii=self.tcp_radius, colors=self.tcp_color)
        self.viewer.log_points("/reach/goal", goal, radii=self.goal_radius, colors=self.goal_color)
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
