"""Check fitted panel transforms and open space above the scoop surface."""

import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "generated_assets/real_tool_demo/geometry.json"
    geometry = json.loads(path.read_text(encoding="utf-8"))
    patches = geometry["patches"]
    centers = np.array([p["center"] for p in patches])
    matrices = np.array([p["rotation"] for p in patches])
    sizes = np.array([p["size"] for p in patches])
    assert np.isfinite(centers).all() and np.isfinite(sizes).all() and (sizes > 0).all()
    np.testing.assert_allclose(matrices.transpose(0, 2, 1) @ matrices,
                               np.broadcast_to(np.eye(3), matrices.shape), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(matrices), 1, atol=1e-8)
    # A point 3 cm on the open side of the fitted floor must not be filled by
    # another panel. This is a geometry check, not a particle-leak/capacity test.
    probes = centers + matrices[:, :, 2] * (sizes[:, 2:3] / 2 + 0.03)
    free = []
    for probe in probes:
        local = np.einsum("nji,nj->ni", matrices, probe - centers)
        free.append(not np.any(np.all(abs(local) <= sizes / 2, axis=1)))
    middle = np.argmin(np.linalg.norm(centers[:, [0, 2]] - [0, 0.39], axis=1))
    assert free[middle], "Center of scoop cavity is incorrectly filled"
    assert sum(free) >= 0.95 * len(free), "Excessive collision intrusion into the cavity"
    out = {"patch_count": len(patches), "open_side_probes": len(free), "free_probes": sum(free),
           "center_probe_free": bool(free[middle]), "particle_leak_tested": False,
           "capacity_2l_validated": False}
    report = root / "logs/asset_audit/collision_check.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("TOOL_COLLISION_GEOMETRY_PASS")


if __name__ == "__main__":
    main()
