"""Validate randomized starts and balanced near/medium/far reachable goals."""

from __future__ import annotations

import argparse

import torch

from .reach_env import ACTION_SCALE_RAD
from .rsl_rl_env import GOAL_DISTANCE_BANDS_M, Ec66RslRlEnv


def run_gate2f_test(*, device: str, num_envs: int) -> None:
    env = Ec66RslRlEnv(
        num_envs,
        device=device,
        random_goals=True,
        random_starts=True,
        seed=17,
    )
    initial_q = env.physics.joint_position.clone()
    nominal_q = env.physics.initial_joint_pos
    posture_delta = torch.linalg.vector_norm(initial_q - nominal_q, dim=-1)
    if torch.any(posture_delta < 1.0e-3):
        raise AssertionError("At least one randomized start is effectively the nominal posture")
    if torch.any(initial_q < env.physics.joint_lower) or torch.any(initial_q > env.physics.joint_upper):
        raise AssertionError("A randomized start violates a joint limit")

    initial_distance = env.previous_distance.clone()
    band_counts: dict[str, int] = {}
    for band_name, (minimum, maximum) in GOAL_DISTANCE_BANDS_M.items():
        in_band = (initial_distance >= minimum - 1.0e-5) & (
            initial_distance <= maximum + 1.0e-5
        )
        band_counts[band_name] = int(in_band.sum().item())
        if band_counts[band_name] == 0:
            raise AssertionError(f"No goals were sampled in the {band_name} distance band")

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
        raise AssertionError(f"Only {int(succeeded.sum())}/{num_envs} randomized cases were reached")

    env.reset()
    changed_start = torch.linalg.vector_norm(env.physics.joint_position - initial_q, dim=-1) > 1.0e-4
    if not torch.all(changed_start):
        raise AssertionError("At least one environment reused its previous initial posture")

    print(f"Randomized environments: {num_envs}")
    print(
        "Initial posture delta range: "
        f"{float(posture_delta.min()):.6f}..{float(posture_delta.max()):.6f} rad"
    )
    print(f"Distance band counts: {band_counts}")
    print(f"All diagnostic cases reached in: {steps_to_finish} steps")
    print("Start posture resampling: verified")
    print("GATE2F_ENV_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=96)
    args = parser.parse_args()
    run_gate2f_test(device=args.device, num_envs=args.num_envs)


if __name__ == "__main__":
    main()
