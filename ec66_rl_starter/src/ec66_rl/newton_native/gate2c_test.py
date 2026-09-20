"""Validate random reachable goal sampling and the batched task loop."""

from __future__ import annotations

import argparse

import torch

from .reach_env import ACTION_SCALE_RAD
from .rsl_rl_env import (
    MAX_INITIAL_GOAL_DISTANCE_M,
    MIN_INITIAL_GOAL_DISTANCE_M,
    Ec66RslRlEnv,
)


def run_gate2c_test(*, device: str, num_envs: int) -> None:
    env = Ec66RslRlEnv(num_envs, device=device, random_goals=True, seed=7)
    first_goals = env.goal_position.clone()
    initial_distance = env.previous_distance.clone()
    if torch.any(initial_distance < MIN_INITIAL_GOAL_DISTANCE_M - 1.0e-5):
        raise AssertionError("A sampled goal is too close to the initial TCP")
    if torch.any(initial_distance > MAX_INITIAL_GOAL_DISTANCE_M + 1.0e-5):
        raise AssertionError("A sampled goal is outside the configured curriculum radius")
    if torch.max(torch.std(first_goals, dim=0)) < 0.02:
        raise AssertionError("Random goals do not have enough spatial diversity")

    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    steps_to_finish = 0
    for step in range(env.max_episode_length):
        error = env.goal_joint_pos - env.physics.joint_position
        actions = (2.0 * error / ACTION_SCALE_RAD).clamp(-1.0, 1.0)
        actions[succeeded] = 0.0
        _, _, _, extras = env.step(actions, auto_reset=False)
        succeeded |= extras["successes"]
        steps_to_finish = step + 1
        if torch.all(succeeded):
            break
    if not torch.all(succeeded):
        raise AssertionError(f"Only {int(succeeded.sum())}/{num_envs} sampled goals were reached")

    env.reset()
    changed = torch.linalg.vector_norm(env.goal_position - first_goals, dim=-1) > 1.0e-4
    if not torch.all(changed):
        raise AssertionError("At least one environment reused its previous random goal")

    print(f"Random goal environments: {num_envs}")
    print(f"Initial distance range: {float(initial_distance.min()):.6f}..{float(initial_distance.max()):.6f} m")
    print(f"All diagnostic goals reached in: {steps_to_finish} steps")
    print("Goal resampling: verified")
    print("GATE2C_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    run_gate2c_test(device=args.device, num_envs=args.num_envs)


if __name__ == "__main__":
    main()
