"""Replay an exported EC66 policy in the Newton viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
import newton.examples

from .reach_task import FixedGoalReachTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def latest_exported_policy() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "logs" / "rsl_rl" / "ec66_reach_fixed").glob("*/exported/policy.pt")
    )
    if not candidates:
        raise FileNotFoundError("No exported EC66 policy found; run train_reach first")
    return candidates[-1]


class Example:
    def __init__(self, viewer, args) -> None:
        self.viewer = viewer
        self.torch_device = torch.device(args.device or "cuda:0")
        self.task = FixedGoalReachTask(device=str(self.torch_device))
        policy_path = Path(args.policy) if args.policy else latest_exported_policy()
        self.policy = torch.jit.load(str(policy_path), map_location=self.torch_device).eval()
        self.policy_path = policy_path
        self.model = self.task.env.model
        self.state_0 = self.task.env.state_0
        self.state_1 = self.task.env.state_1
        self.control = self.task.env.control
        self.contacts = self.task.env.contacts
        self.sim_time = 0.0
        self.pause_after_success = 0
        self.success_count = 0
        self.minimum_distance_m = self.task.previous_distance_m
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
            obs = torch.from_numpy(self.task.observation()).to(self.torch_device).unsqueeze(0)
            with torch.inference_mode():
                action = self.policy(obs).squeeze(0).cpu().numpy().astype(np.float32)
            result = self.task.step(action)
            self.minimum_distance_m = min(self.minimum_distance_m, result.distance_m)
            if result.succeeded:
                self.success_count += 1
                self.pause_after_success = 30
            elif result.truncated:
                self.task.reset()
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

    def test_final(self) -> None:
        if self.success_count < 1:
            raise AssertionError(
                f"Exported policy did not reach the goal; minimum distance={self.minimum_distance_m:.6f} m"
            )
        print(f"EC66_POLICY={self.policy_path}")
        print(f"EC66_REPLAY_SUCCESSES={self.success_count}")
        print(f"EC66_REPLAY_MIN_DISTANCE_M={self.minimum_distance_m:.6f}")
        print("EC66_EXPORTED_POLICY_REPLAY_PASS")


def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--policy", default=None, help="exported TorchScript policy; defaults to latest")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
