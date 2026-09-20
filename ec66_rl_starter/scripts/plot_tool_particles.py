"""Render diagnostic particle snapshots and collision surfaces (not GL QA)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ec66_rl.newton_native.tool_particles import tool_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    data = np.load(args.snapshot)
    fig = plt.figure(figsize=(12, 5))
    for index, key in enumerate(("initial", "final"), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        faces = []
        for center, rotation, size in tool_geometry():
            corners = np.array([[-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * size/2
            faces.append(corners @ rotation.T + center)
        ax.add_collection3d(Poly3DCollection(faces, facecolor="#7a9ba8", edgecolor="#456070", linewidth=.3, alpha=.7))
        q = data[key]
        ax.scatter(*q.T, s=3, color="#cb8f28", alpha=.7)
        ax.set(xlim=(-.2,.2), ylim=(-.2,.2), zlim=(0,.85), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax.set_title(f"{key}: {len(q)} beads; provisional blade")
        ax.set_box_aspect((1,1,1.4))
    fig.suptitle("Static XPBD contact diagnostic - not calibrated sand or measured 2 L capacity")
    fig.tight_layout()
    output = args.snapshot.with_suffix(".png")
    fig.savefig(output, dpi=140)
    print(output)


if __name__ == "__main__":
    main()
