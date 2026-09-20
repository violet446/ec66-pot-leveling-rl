"""Replay the exported random-goal EC66 PPO policy in the Newton viewer."""

from __future__ import annotations

from pathlib import Path

import torch
import warp as wp

import newton
import newton.examples

from .rsl_rl_env import Ec66RslRlEnv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def latest_exported_policy() -> Path:
    for experiment_name in ("ec66_reach_robust", "ec66_reach_random"):
        candidates = sorted(
            (PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name).glob(
                "*/exported/policy.pt"
            )
        )
        if candidates:
            return candidates[-1]
    raise FileNotFoundError("No random-goal EC66 policy found; run train_reach first")


class Example:
    def __init__(self, viewer, args) -> None:
        self.viewer = viewer
        self.torch_device = torch.device(args.device or "cuda:0")
        self.random_starts = not args.fixed_start
        self.env = Ec66RslRlEnv(
            1,
            device=str(self.torch_device),
            random_goals=True,
            random_starts=self.random_starts,
            seed=args.goal_seed,
        )
        policy_path = Path(args.policy) if args.policy else latest_exported_policy()
        self.policy = torch.jit.load(str(policy_path), map_location=self.torch_device).eval()
        self.policy_path = policy_path
        self.model = self.env.physics.model
        self.state_0 = self.env.physics.state_0
        self.state_1 = self.env.physics.state_1
        self.control = self.env.physics.control
        self.contacts = self.env.physics.contacts
        self.sim_time = 0.0
        self.pause_after_success = 0
        self.success_count = 0
        self.goal_count = 1
        self.minimum_distance_m = float(self.env.previous_distance[0].item())
        self.tcp_radius = wp.array([0.025], dtype=wp.float32, device=self.model.device)
        self.goal_radius = wp.array([0.04], dtype=wp.float32, device=self.model.device)
        self.tcp_color = wp.array([(1.0, 0.2, 0.2)], dtype=wp.vec3, device=self.model.device)
        self.goal_color = wp.array([(0.2, 1.0, 0.2)], dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.8, -2.5, 1.5), pitch=-18.0, yaw=145.0)

    def _reset_with_new_goal(self) -> None:
        self.env.reset()
        self.goal_count += 1

    def step(self) -> None:
        if self.pause_after_success > 0:
            self.pause_after_success -= 1
            if self.pause_after_success == 0:
                self._reset_with_new_goal()
        else:
            with torch.inference_mode():
                action = self.policy(self.env._observations_tensor())
                _, _, dones, extras = self.env.step(action, auto_reset=False)
            distance_m = float(extras["distance"][0].item())
            self.minimum_distance_m = min(self.minimum_distance_m, distance_m)
            if bool(extras["successes"][0].item()):
                self.success_count += 1
                self.pause_after_success = 30
            elif bool(dones[0].item()):
                self._reset_with_new_goal()
        self.state_0 = self.env.physics.state_0
        self.state_1 = self.env.physics.state_1
        self.sim_time += self.env.cfg["control_dt_s"]

    def render(self) -> None:
        tcp_position = self.env.physics.tcp_position[0].detach().cpu().tolist()
        goal_position = self.env.goal_position[0].detach().cpu().tolist()
        tcp = wp.array([tcp_position], dtype=wp.vec3, device=self.model.device)
        goal = wp.array([goal_position], dtype=wp.vec3, device=self.model.device)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.physics.state_0)
        self.viewer.log_points("/reach/tcp", tcp, radii=self.tcp_radius, colors=self.tcp_color)
        self.viewer.log_points("/reach/goal", goal, radii=self.goal_radius, colors=self.goal_color)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.success_count < 1:
            raise AssertionError(
                "Random-goal policy did not reach a goal; "
                f"minimum distance={self.minimum_distance_m:.6f} m"
            )
        print(f"EC66_POLICY={self.policy_path}")
        print(f"EC66_RANDOM_STARTS={self.random_starts}")
        print(f"EC66_RANDOM_GOALS_SHOWN={self.goal_count}")
        print(f"EC66_RANDOM_REPLAY_SUCCESSES={self.success_count}")
        print(f"EC66_RANDOM_REPLAY_MIN_DISTANCE_M={self.minimum_distance_m:.6f}")
        print("EC66_RANDOM_POLICY_REPLAY_PASS")


def main() -> None:
    parser = newton.examples.create_parser()
    parser.add_argument("--policy", default=None, help="exported TorchScript policy; defaults to latest")
    parser.add_argument("--goal-seed", type=int, default=2026, help="random goal sequence seed")
    parser.add_argument(
        "--fixed-start",
        action="store_true",
        help="keep the old fixed initial posture instead of Gate 2F random starts",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
