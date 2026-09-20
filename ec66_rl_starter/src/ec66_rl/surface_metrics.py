"""Settled surface measurements and per-throw redistribution reward contract.

Independent of the particle solver. All heights are world Z in meters.
Unobserved cells remain NaN and must not be silently discarded from rewards.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurfaceGrid:
    center_xy_m: tuple[float, float]
    radius_m: float
    resolution: int = 40

    def __post_init__(self):
        if not np.isfinite(self.radius_m) or self.radius_m <= 0 or self.resolution < 4:
            raise ValueError("Invalid surface grid")

    @property
    def centers(self):
        axis = ((np.arange(self.resolution) + 0.5) / self.resolution * 2 - 1) * self.radius_m
        x, y = np.meshgrid(axis + self.center_xy_m[0], axis + self.center_xy_m[1])
        return x, y

    @property
    def mask(self):
        x, y = self.centers
        return (x - self.center_xy_m[0]) ** 2 + (y - self.center_xy_m[1]) ** 2 <= self.radius_m ** 2


def heightmap_from_particles(positions_m, grid: SurfaceGrid, particle_radius_m=0.0):
    """Point-bin top envelope; sparse cells are unknown, not empty floor.

    This is a baseline sensor, not an MPM surface reconstruction. Choose grid
    size compatible with particle spacing or supply a reconstructed heightmap.
    Caller must ensure particles have settled before using its result as reward.
    """
    p = np.asarray(positions_m, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("positions must be finite (N,3)")
    if not np.isfinite(particle_radius_m) or particle_radius_m < 0:
        raise ValueError("Invalid particle radius")
    delta = p[:, :2] - grid.center_xy_m
    inside = (delta ** 2).sum(axis=1) < grid.radius_m ** 2
    ij = np.floor((delta[inside] / grid.radius_m + 1) * grid.resolution / 2).astype(int)
    heights = np.full((grid.resolution, grid.resolution), -np.inf)
    np.maximum.at(heights, (ij[:, 1], ij[:, 0]), p[inside, 2] + particle_radius_m)
    heights[~np.isfinite(heights) | ~grid.mask] = np.nan
    return heights


def surface_statistics(heights_m, grid: SurfaceGrid):
    h = np.asarray(heights_m, dtype=float)
    if h.shape != grid.mask.shape:
        raise ValueError("Heightmap shape does not match the grid")
    observed = grid.mask & np.isfinite(h)
    coverage = float(observed.sum() / grid.mask.sum())
    if coverage < 1.0:
        raise ValueError(f"Incomplete surface coverage {coverage:.3f}; reconstruct missing cells first")
    z = h[grid.mask]
    return {"mean_height_m": float(z.mean()),
            "rms_roughness_m": float(np.sqrt(np.mean((z - z.mean()) ** 2))),
            "p95_p05_m": float(np.quantile(z, 0.95) - np.quantile(z, 0.05)),
            "peak_to_valley_m": float(z.max() - z.min()), "coverage": coverage}


def redistribution_reward(before_m, after_m, grid: SurfaceGrid, *, settled: bool,
                          loaded_volume_m3: float, target_volume_m3: float,
                          spilled_volume_m3: float, residual_volume_m3: float,
                          volume_balance_error_m3: float, wall_impulse_ns: float = 0.0):
    """Draft dimensionless reward for an entire scoop-return cycle.

    Baseline is BEFORE scooping; final measurement is AFTER returning/settling.
    All weights and tolerances are provisional. Volume uses a consistent bulk
    volume convention; mass conservation should be checked in the solver too.
    """
    if not settled:
        raise ValueError("Reward requires a settled surface, not airborne particles")
    values = [loaded_volume_m3, spilled_volume_m3, residual_volume_m3, wall_impulse_ns]
    if target_volume_m3 <= 0 or not np.isfinite(target_volume_m3) or not all(
            np.isfinite(v) and v >= 0 for v in values) or not np.isfinite(volume_balance_error_m3):
        raise ValueError("Invalid volume/impulse measurements")
    before = surface_statistics(before_m, grid)
    after = surface_statistics(after_m, grid)
    terms = {
        "flatness_gain": (before["rms_roughness_m"] - after["rms_roughness_m"]) / 0.05,
        "remaining_roughness": -0.2 * after["rms_roughness_m"] / 0.05,
        "spill": -5.0 * spilled_volume_m3 / target_volume_m3,
        "residual": -residual_volume_m3 / target_volume_m3,
        "dose_error": -abs(loaded_volume_m3 - target_volume_m3) / target_volume_m3,
        "balance_error": -5.0 * abs(volume_balance_error_m3) / target_volume_m3,
        "wall_impulse": -wall_impulse_ns / 10.0,
    }
    return {"reward": float(sum(terms.values())), "terms": terms, "after": after,
            "dose_valid": abs(loaded_volume_m3 - target_volume_m3) <= 0.05 * target_volume_m3}
