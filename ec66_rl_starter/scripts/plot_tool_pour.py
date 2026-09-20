"""Scientific snapshot QA using simulated tool poses and particle states."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import warp as wp
from ec66_rl.newton_native.tool_particles import tool_frame
from ec66_rl.newton_native.scene_builder import PROJECT_ROOT


def main():
    folder = PROJECT_ROOT / "logs/tool_pour"
    snapshots = np.load(folder / "stages.npz")
    data, rotation, translation = tool_frame()
    mesh = trimesh.load(str(PROJECT_ROOT.parent / "assets/铁锹4y向上.STL"), force="mesh", process=False)
    local = (np.asarray(mesh.vertices)*data["config"]["stl_scale_to_m"]
             -data["config"]["mount_center_raw_m"]) @ rotation.T + translation - [0,0,.5]
    fig = plt.figure(figsize=(14, 4))
    for index, (t,title) in enumerate(((20,"Loaded"),(24,"Carried"),(27,"Tipped"),(37,"Settled")), 1):
        ax = fig.add_subplot(1,4,index,projection="3d")
        transform = snapshots[f"tool_{t}"]
        r = np.array(wp.quat_to_matrix(wp.quat(*transform[3:]))).reshape(3,3)
        vertices = local @ r.T + transform[:3]
        ax.add_collection3d(Poly3DCollection(vertices[mesh.faces], facecolor="#91a8b5", linewidth=0, alpha=.5))
        q = snapshots[f"particles_{t}"]
        ax.scatter(*q.T, s=2, color="#b77a13", depthshade=False)
        a = np.linspace(0,2*np.pi,100)
        ax.plot(.2+.4*np.cos(a), .4*np.sin(a), np.zeros(100), color="green", linewidth=1)
        ax.set(xlim=(-.25,.8), ylim=(-.5,.5), zlim=(0,1), title=f"{title} (t={t}s)")
        ax.view_init(elev=23,azim=-65)
        ax.set_box_aspect((1,1,1))
    fig.suptitle("Prescribed scoop + coarse beads | no robot / PPO / two-way coupling")
    fig.tight_layout()
    fig.savefig(folder / "stages.png", dpi=140)
    print(folder / "stages.png")


if __name__ == "__main__":
    main()
