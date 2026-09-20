import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ec66_rl.reach_contract import (  # noqa: E402
    ReachTaskConfig,
    action_to_joint_target,
    build_observation,
    compute_reward,
)


class ReachContractTests(unittest.TestCase):
    def setUp(self):
        self.config = ReachTaskConfig(
            joint_lower=np.full(6, -1.0),
            joint_upper=np.full(6, 1.0),
        )

    def test_observation_has_documented_size_and_goal_delta(self):
        observation = build_observation(
            self.config,
            joint_pos=np.zeros(6),
            joint_vel=np.zeros(6),
            tcp_pos=np.array([0.1, 0.2, 0.3]),
            goal_pos=np.array([0.4, 0.2, 0.3]),
            previous_action=np.zeros(6),
        )
        self.assertEqual(observation.shape, (22,))
        np.testing.assert_allclose(observation[12:15], [0.3, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(observation[-1]), 0.3, places=6)

    def test_action_is_scaled_and_clipped_to_joint_limits(self):
        target = action_to_joint_target(
            self.config,
            joint_pos=np.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0]),
            action=np.array([3.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(target, [1.0, -0.04, 0.02, 0.0, 0.0, 0.0], atol=1e-6)

    def test_success_is_better_than_no_progress(self):
        ordinary_reward, ordinary_done, ordinary_success = compute_reward(
            self.config, 0.20, 0.19, np.zeros(6)
        )
        success_reward, success_done, success = compute_reward(self.config, 0.04, 0.02, np.zeros(6))
        self.assertFalse(ordinary_done)
        self.assertFalse(ordinary_success)
        self.assertTrue(success_done)
        self.assertTrue(success)
        self.assertGreater(success_reward, ordinary_reward)


if __name__ == "__main__":
    unittest.main()
