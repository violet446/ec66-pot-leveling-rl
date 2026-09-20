import unittest
import numpy as np
from ec66_rl.surface_metrics import SurfaceGrid, heightmap_from_particles, surface_statistics, redistribution_reward


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.grid = SurfaceGrid((0, 0), 0.97, 40)
        self.flat = np.full((40, 40), -0.3)
        x, _ = self.grid.centers
        self.tilt = self.flat + 0.1 * x

    def reward(self, before, after, **extra):
        kwargs = dict(settled=True, loaded_volume_m3=0.002, target_volume_m3=0.002,
                      spilled_volume_m3=0, residual_volume_m3=0, volume_balance_error_m3=0)
        kwargs.update(extra)
        return redistribution_reward(before, after, self.grid, **kwargs)

    def test_horizontal_translation_and_slope(self):
        self.assertLess(surface_statistics(self.flat, self.grid)["rms_roughness_m"], 1e-12)
        self.assertGreater(surface_statistics(self.tilt, self.grid)["rms_roughness_m"], 0.04)
        self.assertAlmostEqual(surface_statistics(self.tilt, self.grid)["rms_roughness_m"],
                               surface_statistics(self.tilt + 0.2, self.grid)["rms_roughness_m"])

    def test_missing_data_and_unsettled_are_rejected(self):
        broken = self.flat.copy()
        broken[20, 20] = np.nan
        with self.assertRaises(ValueError):
            self.reward(self.tilt, broken)
        with self.assertRaises(ValueError):
            self.reward(self.tilt, self.flat, settled=False)

    def test_improvement_and_spill_penalty(self):
        good = self.reward(self.tilt, self.flat)
        self.assertGreater(good["reward"], self.reward(self.tilt, self.tilt)["reward"])
        self.assertLess(self.reward(self.tilt, self.flat, spilled_volume_m3=0.002)["reward"], 0)
        self.assertFalse(self.reward(self.tilt, self.flat, loaded_volume_m3=0)["dose_valid"])

    def test_particle_envelope_and_outside_mask(self):
        x, y = self.grid.centers
        p = np.column_stack((x[self.grid.mask], y[self.grid.mask], self.tilt[self.grid.mask]))
        p = np.concatenate((p, p - [0, 0, 0.1], [[3, 3, 100]]))
        h = heightmap_from_particles(p, self.grid)
        np.testing.assert_allclose(h[self.grid.mask], self.tilt[self.grid.mask])
        with self.assertRaises(ValueError):
            surface_statistics(heightmap_from_particles(p[:1], self.grid), self.grid)


if __name__ == "__main__":
    unittest.main()
