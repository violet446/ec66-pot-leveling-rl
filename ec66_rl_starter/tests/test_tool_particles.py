import unittest
import numpy as np
from ec66_rl.newton_native.tool_particles import upper_surface, tool_geometry, ToolParticleScene


class ToolGeometryTests(unittest.TestCase):
    def test_invalid_parameters_rejected_before_build(self):
        for options in ({"radius": 0}, {"dose_liters": -1}, {"substeps": 3}, {"relaxation": 0}):
            with self.assertRaises(ValueError):
                ToolParticleScene(**options)

    def test_ray_box_height_and_missing_surface(self):
        patches = [(np.array([0., 0., .5]), np.eye(3), np.array([.2, .2, .02]))]
        result = upper_surface([[0, 0], [.11, 0]], patches)
        self.assertAlmostEqual(result[0], .51)
        self.assertTrue(np.isnan(result[1]))

    def test_rotated_box(self):
        angle = .4
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        result = upper_surface([[0, 0]], [(np.array([0., 0., .5]), rotation, np.array([.2, .2, .02]))])
        self.assertAlmostEqual(result[0], .5 + .01/c)

    def test_actual_patch_transforms(self):
        patches = tool_geometry()
        self.assertEqual(len(patches), 135)
        for center, rotation, size in patches:
            self.assertTrue(np.isfinite(center).all())
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertTrue((size > 0).all())


if __name__ == "__main__":
    unittest.main()
