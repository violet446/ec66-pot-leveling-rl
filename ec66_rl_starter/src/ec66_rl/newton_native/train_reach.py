"""Train, evaluate, and export an EC66 Reach PPO policy."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from .rsl_rl_env import Ec66RslRlEnv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def make_train_cfg(iterations: int, experiment_name: str) -> dict:
    return {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": iterations,
        "empirical_normalization": {},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": 1.0,
        "check_for_nan": True,
        "save_interval": max(10, iterations // 5),
        "experiment_name": experiment_name,
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "class_name": "OnPolicyRunner",
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {},
    }


def evaluate(runner: OnPolicyRunner, env: Ec66RslRlEnv, episodes: int) -> tuple[int, int]:
    policy = runner.get_inference_policy(device=str(env.device))
    env.reset()
    successes = 0
    completed = 0
    with torch.inference_mode():
        while completed < episodes:
            actions = policy(env.get_observations())
            _, _, dones, extras = env.step(actions)
            if torch.any(dones):
                done_successes = extras["successes"][dones.bool()]
                count = min(episodes - completed, int(done_successes.numel()))
                successes += int(done_successes[:count].sum().item())
                completed += count
    return successes, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--fixed-goal", action="store_true", help="retain the Gate 2B fixed target")
    parser.add_argument(
        "--random-starts",
        action="store_true",
        help="randomize the safe initial joint posture for Gate 2F",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.fixed_goal:
        experiment_name = "ec66_reach_fixed"
    elif args.random_starts:
        experiment_name = "ec66_reach_robust"
    else:
        experiment_name = "ec66_reach_random"
    log_dir = PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Ec66RslRlEnv(
        args.num_envs,
        device=args.device,
        random_goals=not args.fixed_goal,
        random_starts=args.random_starts,
        seed=42,
    )
    train_cfg = make_train_cfg(args.iterations, experiment_name)
    train_cfg["device"] = args.device
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    checkpoint = log_dir / f"model_{args.iterations}.pt"
    runner.save(str(checkpoint))
    export_dir = log_dir / "exported"
    export_dir.mkdir(exist_ok=True)
    runner.export_policy_to_jit(str(export_dir), filename="policy.pt")
    runner.export_policy_to_onnx(str(export_dir), filename="policy.onnx")
    successes, completed = evaluate(runner, env, episodes=args.eval_episodes)

    print(f"EC66_LOG_DIR={log_dir}")
    print(f"EC66_CHECKPOINT={checkpoint}")
    print(f"EC66_EVAL_SUCCESS={successes}/{completed}")
    success_rate = successes / max(completed, 1)
    print(f"EC66_EVAL_SUCCESS_RATE={success_rate:.4f}")
    print("EC66_RL_PIPELINE_PASS" if success_rate >= 0.8 else "EC66_RL_PIPELINE_NEEDS_TUNING")


if __name__ == "__main__":
    main()
