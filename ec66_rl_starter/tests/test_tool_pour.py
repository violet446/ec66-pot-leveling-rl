import unittest
import numpy as np
from ec66_rl.newton_native.tool_pour import pour_phase
from ec66_rl.newton_native.tool_particles import tool_geometry, tool_frame


class PourContractTests(unittest.TestCase):
    def test_stage_boundaries(self):
        self.assertEqual([pour_phase(t) for t in (0,19,20,24,27,37)],
                         ["LOAD_SETTLE", "LOAD_SETTLE", "CARRY", "TIP", "LAND_SETTLE", "DONE"])

    def test_visual_and_collision_share_frame(self):
        data, rotation, translation = tool_frame()
        patches = tool_geometry()
        for source, (center, r, size) in zip(data["patches"], patches):
            np.testing.assert_allclose(center, rotation @ source["center"] + translation)
            np.testing.assert_allclose(r, rotation @ source["rotation"])


if __name__ == "__main__":
    unittest.main()
