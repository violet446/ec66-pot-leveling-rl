import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ec66_trajectory import (  # noqa: E402
    BLENDER_FPS,
    JOINT_SPEED_LIMITS_DEG_S,
    build_reference_trajectory,
    forward_kinematics,
    target_at_frame,
)


def test_reference_trajectory_matches_tool_targets():
    trajectory = build_reference_trajectory()
    errors = []
    for time_seconds, joint_q, tool in zip(trajectory.times, trajectory.joint_q, trajectory.tool_pose, strict=True):
        frame = time_seconds * BLENDER_FPS + 1.0
        target = target_at_frame(frame)
        errors.append(np.linalg.norm(tool[:3, 3] - np.array((target[0], 0.0, target[1]))))
        np.testing.assert_allclose(forward_kinematics(joint_q)[0], tool, atol=1.0e-12)
    assert max(errors) < 1.0e-5


def test_reference_joint_speed_limits():
    trajectory = build_reference_trajectory()
    dt = np.diff(trajectory.times)
    speeds_deg_s = np.max(np.abs(np.diff(trajectory.joint_q, axis=0) / dt[:, None]), axis=0) * 180.0 / np.pi
    assert np.all(speeds_deg_s <= JOINT_SPEED_LIMITS_DEG_S)

