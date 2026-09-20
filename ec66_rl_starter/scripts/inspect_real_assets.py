"""Read-only asset audit; write diagnostic bounds and orthographic mesh views."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root.parent / "assets"
    out = root / "logs" / "asset_audit"
    out.mkdir(parents=True, exist_ok=True)
    urdf = ET.parse(assets / "ec66_description.urdf").getroot()
    report = {"mesh_references": [], "meshes": [], "scoop_scale_assumption": 0.001}
    for ref in urdf.findall(".//mesh"):
        path = assets / ref.get("filename")
        report["mesh_references"].append({"path": str(path), "exists": path.is_file()})
    for path in sorted((assets / "ec66").glob("*.STL")) + sorted(assets.glob("*.STL")):
        mesh = trimesh.load(str(path), force="mesh", process=False)
        report["meshes"].append({"path": str(path), "triangles": len(mesh.faces),
            "bounds_raw": None if mesh.bounds is None else mesh.bounds.tolist()})
        if path.parent != assets:
            continue
        vertices = np.asarray(mesh.vertices) * 0.001
        fig, axes = plt.subplots(1, 3, figsize=(13, 6))
        for ax, (a, b, title) in zip(axes, [(0, 2, "X-Z"), (1, 2, "Y-Z"), (0, 1, "X-Y")]):
            triangles = vertices[mesh.faces][:, :, [a, b]]
            ax.add_collection(PolyCollection(triangles, facecolors=(0.35, 0.55, 0.72, 0.16),
                                             edgecolors=(0.1, 0.2, 0.3, 0.10), linewidths=0.15))
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_xlabel(f"{'XYZ'[a]} (m, assumed STL scale 0.001)")
            ax.set_ylabel(f"{'XYZ'[b]} (m)")
            ax.set_title(title)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "scoop_orthographic.png", dpi=150)
        plt.close(fig)
        # Cross-section bounds expose handle/blade transition without assuming origin.
        report["scoop_sections"] = []
        for lo in np.arange(0, 0.55, 0.05):
            section = vertices[(vertices[:, 2] >= lo) & (vertices[:, 2] < lo + 0.05)]
            if len(section):
                report["scoop_sections"].append({"z_m": float(lo), "min_xy": section[:, :2].min(0).tolist(),
                                                 "max_xy": section[:, :2].max(0).tolist()})
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["scoop_sections"], indent=2))
    print(out / "scoop_orthographic.png")


if __name__ == "__main__":
    main()
