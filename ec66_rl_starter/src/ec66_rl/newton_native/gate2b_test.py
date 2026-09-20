"""Gate 2B acceptance test for the fixed-goal Reach-v0 task loop."""

from __future__ import annotations

import argparse

import numpy as np

from .reach_task import EPISODE_LENGTH_STEPS, FixedGoalReachTask


def run_gate2b_test(*, device: str) -> None:
    task = FixedGoalReachTask(device=device)
    initial_observation = task.observation()
    if initial_observation.shape != (22,):
        raise AssertionError(f"Expected a 22D observation, got {initial_observation.shape}")
    if not np.isfinite(initial_observation).all():
        raise AssertionError("Initial observation contains NaN or infinity")

    initial_distance = task.previous_distance_m
    total_reward = 0.0
    final_step = None
    for _ in range(EPISODE_LENGTH_STEPS):
        final_step = task.step(task.joint_space_oracle_action())
        total_reward += final_step.reward
        if final_step.terminated or final_step.truncated:
            break

    if final_step is None or not final_step.succeeded:
        final_distance = float("nan") if final_step is None else final_step.distance_m
        raise AssertionError(
            f"Diagnostic controller did not reach the fixed goal: initial={initial_distance:.6f} m, "
            f"final={final_distance:.6f} m"
        )
    if final_step.observation.shape != (22,) or not np.isfinite(final_step.observation).all():
        raise AssertionError("Final observation is invalid")

    steps_to_success = task.episode_step
    reset_observation = task.reset()
    if task.episode_step != 0 or reset_observation.shape != (22,):
        raise AssertionError("Task reset did not restore the episode contract")

    print(f"Scoop cutting-edge goal: {np.array2string(task.goal_position, precision=6)}")
    print(f"Initial distance: {initial_distance:.6f} m")
    print(f"Success distance: {final_step.distance_m:.6f} m")
    print(f"Steps to success: {steps_to_success}")
    print(f"Episode return before reset: {total_reward:.6f}")
    print("Observation shape: (22,)")
    print("Reset: verified")
    print("GATE2B_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_gate2b_test(device=args.device)


if __name__ == "__main__":
    main()
