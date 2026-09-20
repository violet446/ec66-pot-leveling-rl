"""Executable Gate 2A acceptance test for the Newton-native EC66 scene."""

from __future__ import annotations

import argparse

import numpy as np

import newton

from .reach_env import Ec66ReachEnv
from .scene_builder import INITIAL_JOINT_POS, JOINT_NAMES


def run_smoke_test(*, device: str, steps: int) -> None:
    if steps < 20:
        raise ValueError("Gate 2A smoke test needs at least 20 control steps")

    env = Ec66ReachEnv(device=device)
    split_step = steps // 2
    initial_tcp = env.tcp_position()
    max_hold_error = 0.0
    max_abs_velocity = 0.0

    for _ in range(split_step):
        env.step()
        env.assert_finite_and_within_limits()
        max_hold_error = max(
            max_hold_error,
            float(np.max(np.abs(env.joint_position() - INITIAL_JOINT_POS))),
        )
        max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(env.joint_velocity()))))

    tcp_before_action = env.tcp_position()
    q_before_action = env.joint_position()
    action = np.zeros(6, dtype=np.float32)
    action[0] = 0.5  # +0.02 rad with the Reach-v0 action scale.
    env.apply_action(action)

    for _ in range(steps - split_step):
        env.step()
        env.assert_finite_and_within_limits()
        max_abs_velocity = max(max_abs_velocity, float(np.max(np.abs(env.joint_velocity()))))

    final_q = env.joint_position()
    final_qd = env.joint_velocity()
    final_tcp = env.tcp_position()
    tcp_displacement = float(np.linalg.norm(final_tcp - tcp_before_action))
    commanded_joint_response = float(final_q[0] - q_before_action[0])
    final_target_error = float(np.max(np.abs(final_q - env.target_joint_pos)))

    if max_hold_error > 0.05:
        raise AssertionError(f"Zero-action hold error is too large: {max_hold_error:.6f} rad")
    if commanded_joint_response < 0.005:
        raise AssertionError(f"Joint 1 did not respond to its +0.02 rad target: {commanded_joint_response:.6f} rad")
    if tcp_displacement < 1.0e-3:
        raise AssertionError(f"TCP did not move after the small action: {tcp_displacement:.6f} m")
    if final_target_error > 0.05:
        raise AssertionError(f"Final joint target error is too large: {final_target_error:.6f} rad")

    print(f"Newton version: {newton.__version__}")
    print(f"Device: {env.model.device}")
    print(f"Joint order: {JOINT_NAMES}")
    print(f"Initial TCP: {np.array2string(initial_tcp, precision=6)}")
    print(f"TCP before action: {np.array2string(tcp_before_action, precision=6)}")
    print(f"Final TCP: {np.array2string(final_tcp, precision=6)}")
    print(f"Final q: {np.array2string(final_q, precision=6)}")
    print(f"Final qd: {np.array2string(final_qd, precision=6)}")
    print(f"Max zero-action hold error: {max_hold_error:.6f} rad")
    print(f"Max absolute joint velocity: {max_abs_velocity:.6f} rad/s")
    print(f"Joint 1 response: {commanded_joint_response:.6f} rad")
    print(f"TCP displacement after action: {tcp_displacement:.6f} m")
    print(f"Final target error: {final_target_error:.6f} rad")
    print("GATE2A_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Warp/Newton device")
    parser.add_argument("--steps", type=int, default=250, help="total 20 Hz control steps")
    args = parser.parse_args()
    run_smoke_test(device=args.device, steps=args.steps)


if __name__ == "__main__":
    main()
