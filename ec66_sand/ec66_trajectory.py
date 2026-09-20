"""Pure-NumPy EC66 reference trajectory extracted from the Blender scene.

This module deliberately has no bpy/mathutils dependency.  Newton and tests can
therefore use exactly the same kinematic reference without launching Blender.
Units are metres, seconds, and radians.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np


BLENDER_FPS = 24.0
END_FRAME = 216.0
BASE_POSITION = np.array([-1.48, -0.122, 0.95], dtype=np.float64)

JOINT_ORIGINS = (
    (0.0, 0.0, 0.096),
    (0.0, 0.122, 0.0),
    (0.0, -0.122, 0.418),
    (0.0, 0.0, 0.398),
    (0.0, 0.122, 0.0),
    (0.0, 0.0, 0.098),
)
ORIGIN_PITCH = (0.0, math.pi / 2.0, 0.0, math.pi / 2.0, 0.0, 0.0)
JOINT_AXES = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
JOINT_LIMITS_DEG = np.array(
    [(-360.0, 360.0), (-360.0, 360.0), (-160.0, 160.0),
     (-360.0, 360.0), (-360.0, 360.0), (-360.0, 360.0)],
    dtype=np.float64,
)
JOINT_SPEED_LIMITS_DEG_S = np.array([150.0, 150.0, 190.0, 260.0, 260.0, 260.0])

# frame, tool target X, tool target Z, tool pitch in degrees
KEYS = (
    (1.0, -0.97, 1.14, 0.0),
    (29.0, -0.95, 1.10, -6.0),
    (54.0, -0.90, 0.94, -20.0),
    (72.0, -0.86, 0.89, -20.0),
    (82.0, -0.82, 0.88, -15.0),
    (96.0, -0.88, 0.90, 8.0),
    (106.0, -0.97, 0.99, 12.0),
    (118.0, -1.00, 1.09, 14.0),
    (131.0, -0.90, 1.21, 14.0),
    (142.0, -0.80, 1.30, -8.0),
    (151.0, -0.76, 1.35, -40.0),
    (162.0, -0.76, 1.36, -43.0),
    (168.0, -0.78, 1.36, -40.0),
    (191.0, -0.99, 1.28, -5.0),
    (216.0, -0.97, 1.14, 0.0),
)


def _translation(position) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = position
    return result


def _rotation(axis, angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    return result


ORIGIN_TRANSFORMS = tuple(
    _translation(position) @ _rotation((0.0, 1.0, 0.0), pitch)
    for position, pitch in zip(JOINT_ORIGINS, ORIGIN_PITCH, strict=True)
)
TOOL_FIXED = (
    _translation((0.0, -0.089, -0.00047661))
    @ _rotation((0.0, 0.0, 1.0), math.pi)
    @ _rotation((1.0, 0.0, 0.0), -math.pi / 2.0)
    @ _rotation((0.0, 1.0, 0.0), -math.pi / 2.0)
)


def forward_kinematics(joint_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tool transform, joint positions, and joint axes in world space."""
    transform = _translation(BASE_POSITION)
    positions = []
    axes = []
    for origin, axis, angle in zip(ORIGIN_TRANSFORMS, JOINT_AXES, joint_q, strict=True):
        transform = transform @ origin
        positions.append(transform[:3, 3].copy())
        axes.append(transform[:3, :3] @ np.asarray(axis))
        transform = transform @ _rotation(axis, float(angle))
    return transform @ TOOL_FIXED, np.asarray(positions), np.asarray(axes)


def _rotation_vector(rotation_matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation_matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_vector = np.array(
        (
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ),
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * skew_vector
    sine = math.sin(angle)
    if abs(sine) < 1.0e-8:
        # This trajectory does not normally enter the pi singularity; the
        # eigenvector fallback keeps failures diagnosable if it ever does.
        values, vectors = np.linalg.eig(rotation_matrix)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return skew_vector * (0.5 * angle / sine)


def target_at_frame(frame: float) -> np.ndarray:
    """Evaluate the monotone cubic Blender tool-space keyframe curve."""
    for index, (left, right) in enumerate(zip(KEYS, KEYS[1:], strict=True)):
        if left[0] <= frame <= right[0]:
            span = right[0] - left[0]
            u = (frame - left[0]) / span

            def slope(key_index: int, component: int) -> float:
                if key_index in (0, 1, len(KEYS) - 1):
                    return 0.0
                before = (KEYS[key_index][component] - KEYS[key_index - 1][component]) / (
                    KEYS[key_index][0] - KEYS[key_index - 1][0]
                )
                after = (KEYS[key_index + 1][component] - KEYS[key_index][component]) / (
                    KEYS[key_index + 1][0] - KEYS[key_index][0]
                )
                return 0.0 if before * after <= 0.0 else 2.0 * before * after / (before + after)

            return np.asarray(
                [
                    (2.0 * u**3 - 3.0 * u**2 + 1.0) * left[component]
                    + (u**3 - 2.0 * u**2 + u) * span * slope(index, component)
                    + (-2.0 * u**3 + 3.0 * u**2) * right[component]
                    + (u**3 - u**2) * span * slope(index + 1, component)
                    for component in range(1, 4)
                ]
            )
    return np.asarray(KEYS[-1][1:], dtype=np.float64)


def solve_ik(x: float, z: float, pitch_deg: float, seed: np.ndarray) -> np.ndarray:
    goal = _rotation((0.0, 1.0, 0.0), -math.radians(pitch_deg))
    goal[:3, 3] = (x, 0.0, z)
    joint_q = np.asarray(seed, dtype=np.float64).copy()
    for _ in range(100):
        tool, positions, axes = forward_kinematics(joint_q)
        orientation_error = _rotation_vector(goal[:3, :3] @ tool[:3, :3].T)
        error = np.concatenate((goal[:3, 3] - tool[:3, 3], orientation_error * 0.3))
        if np.linalg.norm(error) < 1.0e-6:
            return joint_q
        jacobian = np.vstack(
            (np.asarray([np.cross(axis, tool[:3, 3] - point) for axis, point in zip(axes, positions)]).T, axes.T * 0.3)
        )
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 4.0e-5, error)
        joint_q += delta * min(1.0, 0.18 / max(float(np.max(np.abs(delta))), 1.0e-9))
    raise RuntimeError(f"EC66 IK did not converge: target={(x, z, pitch_deg)}, residual={np.linalg.norm(error):.3e}")


@dataclass(frozen=True)
class ReferenceTrajectory:
    times: np.ndarray
    joint_q: np.ndarray
    tool_pose: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, time_seconds: float, loop: bool = True) -> np.ndarray:
        if loop and self.duration > 0.0:
            time_seconds = time_seconds % self.duration
        else:
            time_seconds = float(np.clip(time_seconds, 0.0, self.duration))
        upper = int(np.searchsorted(self.times, time_seconds, side="right"))
        upper = min(max(upper, 1), len(self.times) - 1)
        lower = upper - 1
        alpha = (time_seconds - self.times[lower]) / (self.times[upper] - self.times[lower])
        return (1.0 - alpha) * self.joint_q[lower] + alpha * self.joint_q[upper]


def build_reference_trajectory(tool_z_offset: Callable[[float], float] | None = None) -> ReferenceTrajectory:
    """Bake half-frame IK samples, optionally offsetting tool Z by frame."""
    frames = np.arange(1.0, END_FRAME + 0.25, 0.5, dtype=np.float64)
    seed_candidates = (
        np.array((0.0, -1.1, 1.7, -1.1, math.pi / 2.0, 0.0)),
        np.array((0.0, -1.8, 1.8, 0.0, math.pi / 2.0, 0.0)),
        np.array((0.0, -1.1, 1.7, 1.0, math.pi / 2.0, math.pi)),
        np.array((0.0, -1.1, 1.7, 1.0, -math.pi / 2.0, 0.0)),
    )
    first_target = target_at_frame(frames[0])
    if tool_z_offset is not None:
        first_target[1] += float(tool_z_offset(float(frames[0])))
    seed = None
    for candidate in seed_candidates:
        try:
            seed = solve_ik(*first_target, candidate)
            break
        except RuntimeError:
            pass
    if seed is None:
        raise RuntimeError("No initial EC66 IK branch could solve the reference pose")

    joint_samples = []
    tool_samples = []
    for frame in frames:
        target = target_at_frame(frame)
        if tool_z_offset is not None:
            target[1] += float(tool_z_offset(float(frame)))
        seed = solve_ik(*target, seed)
        joint_samples.append(seed.copy())
        tool_samples.append(forward_kinematics(seed)[0])

    joint_q = np.asarray(joint_samples)
    lower = np.radians(JOINT_LIMITS_DEG[:, 0])
    upper = np.radians(JOINT_LIMITS_DEG[:, 1])
    if np.any(joint_q < lower) or np.any(joint_q > upper):
        raise RuntimeError("Baked EC66 trajectory violates a joint limit")
    times = (frames - 1.0) / BLENDER_FPS
    return ReferenceTrajectory(times=times, joint_q=joint_q, tool_pose=np.asarray(tool_samples))
