"""Evaluate an exported EC66 PPO policy on fresh random reachable goals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--random-starts", action="store_true")
    parser.add_argument("--distance-band", choices=("near", "medium", "far"), default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    policy_path = Path(args.policy) if args.policy else latest_exported_policy()
    device = torch.device(args.device)
    policy = torch.jit.load(str(policy_path), map_location=device).eval()
    env = Ec66RslRlEnv(
        args.num_envs,
        device=args.device,
        random_goals=True,
        random_starts=args.random_starts,
        goal_distance_band=args.distance_band,
        seed=args.seed,
    )

    successes = 0
    completed = 0
    final_distances: list[torch.Tensor] = []
    with torch.inference_mode():
        while completed < args.episodes:
            actions = policy(env._observations_tensor())
            _, _, dones, extras = env.step(actions)
            if torch.any(dones):
                done_mask = dones.bool()
                done_successes = extras["successes"][done_mask]
                done_distances = extras["distance"][done_mask]
                count = min(args.episodes - completed, int(done_successes.numel()))
                successes += int(done_successes[:count].sum().item())
                final_distances.append(done_distances[:count].detach().cpu())
                completed += count

    distances = torch.cat(final_distances)
    success_rate = successes / completed
    print(f"EC66_POLICY={policy_path}")
    print(f"EC66_RANDOM_STARTS={args.random_starts}")
    print(f"EC66_DISTANCE_BAND={args.distance_band or 'mixed'}")
    print(f"EC66_EVAL_SUCCESS={successes}/{completed}")
    print(f"EC66_EVAL_SUCCESS_RATE={success_rate:.4f}")
    print(f"EC66_EVAL_MEAN_FINAL_DISTANCE_M={distances.mean().item():.6f}")
    print(f"EC66_EVAL_MAX_FINAL_DISTANCE_M={distances.max().item():.6f}")
    print("EC66_RANDOM_POLICY_EVAL_PASS" if success_rate >= 0.8 else "EC66_RANDOM_POLICY_EVAL_FAIL")


if __name__ == "__main__":
    main()
