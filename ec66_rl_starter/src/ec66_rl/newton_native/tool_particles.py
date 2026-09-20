"""Static real-tool XPBD bead diagnostic; not calibrated sand or robot coupling.

Dose uses sphere solid volume / assumed packing fraction. It is a nominal
bulk-volume equivalent, not a measured settled scoop volume.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
import newton

from .scene_builder import PROJECT_ROOT, INITIAL_JOINT_POS
from .pot_motion import ArmKinematics, axis_rotation
from .real_tool import real_tool_paths


def tool_frame():
    urdf, _ = real_tool_paths()
    data = json.loads((urdf.parent / "geometry.json").read_text(encoding="utf-8"))
    _, rotation = ArmKinematics(urdf).pose(INITIAL_JOINT_POS)
    root = ET.parse(urdf).getroot()
    for name in ("flange_joint", "scoop_mount"):
        r, p, y = np.fromstring(root.find(f"joint[@name='{name}']/origin").get("rpy", "0 0 0"), sep=" ")
        rotation = rotation @ axis_rotation([0, 0, 1], y) @ axis_rotation([0, 1, 0], p) @ axis_rotation([1, 0, 0], r)
    # Same orientation as the motion demo; translated onto an isolated test bench.
    centers = np.array([v["center"] for v in data["patches"]]) @ rotation.T
    translation = np.array([0., 0., .5]) - np.median(centers, axis=0)
    return data, rotation, translation


def tool_geometry():
    data, rotation, translation = tool_frame()
    patches = [(rotation @ np.array(v["center"]) + translation,
                rotation @ np.array(v["rotation"]), np.array(v["size"])) for v in data["patches"]]
    return patches


def upper_surface(xy, patches):
    """Vertical ray / oriented-box slabs. NaN means no blade underneath."""
    xy = np.asarray(xy).reshape(-1, 2)
    top = np.full(len(xy), -np.inf)
    for center, rotation, size in patches:
        origin = (np.column_stack((xy, np.zeros(len(xy)))) - center) @ rotation
        direction = rotation[2, :]
        lo, hi = np.full(len(xy), -np.inf), np.full(len(xy), np.inf)
        for axis in range(3):
            if abs(direction[axis]) < 1e-10:
                hi[abs(origin[:, axis]) > size[axis] / 2] = -np.inf
            else:
                a = (-size[axis]/2 - origin[:, axis]) / direction[axis]
                b = (size[axis]/2 - origin[:, axis]) / direction[axis]
                lo = np.maximum(lo, np.minimum(a, b))
                hi = np.minimum(hi, np.maximum(a, b))
        top = np.maximum(top, np.where(lo <= hi, hi, -np.inf))
    return np.where(np.isfinite(top), top, np.nan)


class ToolParticleScene:
    def __init__(self, device="cuda:0", mode="dose", radius=.006, dose_liters=2.,
                 packing_fraction=.60, substeps=32, duration=20., relaxation=.1):
        if mode not in ("dose", "probe") or not 0 < radius <= .02:
            raise ValueError("Invalid mode or radius")
        if not 0 < packing_fraction < 1 or dose_liters <= 0 or substeps < 2 or substeps % 2 or duration <= 0:
            raise ValueError("Invalid dose, packing, duration or substeps")
        self.mode, self.radius, self.packing = mode, radius, packing_fraction
        self.substeps, self.duration = substeps, duration
        if not 0 < relaxation <= 1:
            raise ValueError("Relaxation must be in (0, 1]")
        self.relaxation = relaxation
        self.sim_time, self.frame_dt = 0., 1/60
        self.patches = tool_geometry()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = newton.ModelBuilder.ShapeConfig(mu=.6)
        self.add_tool(builder, cfg)
        # Catch tray is separated from the blade and cannot help it retain beads.
        builder.add_shape_box(body=-1, xform=wp.transform((0., 0., -.025), wp.quat_identity()),
                              hx=1., hy=1., hz=.025, cfg=cfg, label="catch_tray")
        spacing = 2.1 * radius
        xy = np.array([(x, y) for x in np.arange(-.08, .081, spacing)
                       for y in np.arange(-.09, .091, spacing)])
        heights = upper_surface(xy, self.patches)
        xy = xy[np.isfinite(heights)]
        heights = heights[np.isfinite(heights)]
        if len(xy) == 0:
            raise RuntimeError("No blade under dose footprint")
        if mode == "probe":
            positions = np.column_stack((xy, heights + radius + .012))
        else:
            count = round(dose_liters * .001 * packing_fraction / (4/3 * math.pi * radius**3))
            if not 1 <= count <= 20000:
                raise ValueError("Particle count outside diagnostic budget (1..20000)")
            positions = np.array([(xy[i % len(xy), 0], xy[i % len(xy), 1],
                                   heights.max() + radius + .012 + (i//len(xy))*spacing) for i in range(count)])
        self.count = len(positions)
        self.nominal_liters = self.count * 4/3 * math.pi * radius**3 / packing_fraction * 1000
        self.initial_positions = positions
        builder.add_particles(pos=[wp.vec3(*p) for p in positions], vel=[wp.vec3(0.)]*self.count,
                              mass=[1600 * self.nominal_liters * .001 / self.count]*self.count,
                              radius=[radius]*self.count)
        self.model = builder.finalize(device=device)
        self.model.particle_mu = .6
        self.model.particle_cohesion = 0.
        self.model.soft_contact_mu = .6
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=8, soft_contact_relaxation=relaxation)
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=radius,
                                                soft_contact_max=self.count * self.model.shape_count)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.below_surface = np.zeros(self.count, dtype=bool)
        self.peak_contacts = 0
        self.tail_speeds = []
        self.tail_rms = []
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def add_tool(self, builder, cfg):
        for i, (center, rotation, size) in enumerate(self.patches):
            builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), wp.quat_from_matrix(wp.mat33(rotation))),
                                  hx=size[0]/2, hy=size[1]/2, hz=size[2]/2, cfg=cfg, label=f"blade/{i}")

    def simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt/self.substeps)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.duration - 1e-8:
            return
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        q, qd = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("Nonfinite particle state")
        height = upper_surface(q[:, :2], self.patches)
        self.below_surface |= np.isfinite(height) & (q[:, 2] < height - 2*self.radius) & (q[:, 2] > .15)
        contacts = int(self.contacts.soft_contact_count.numpy()[0])
        self.peak_contacts = max(self.peak_contacts, contacts)
        if contacts > self.pipeline.soft_contact_max:
            raise RuntimeError("Soft contact buffer overflow")
        if self.sim_time > self.duration - .5:
            self.tail_speeds.append(float(np.max(np.linalg.norm(qd, axis=1))))
            self.tail_rms.append(float(np.sqrt(np.mean(np.sum(qd*qd, axis=1)))))

    def report(self):
        q = self.state_0.particle_q.numpy()
        height = upper_surface(q[:, :2], self.patches)
        retained = np.isfinite(height) & (q[:, 2] >= height-self.radius) & (q[:, 2] > .2)
        settled = bool(self.tail_speeds and max(self.tail_speeds) < .05)
        ratio = float(retained.mean())
        from scipy.spatial import cKDTree
        nearest = cKDTree(q).query(q, k=2)[0][:, 1]
        overlap = float(max(0., 2*self.radius-nearest.min())) if self.count > 1 else 0.
        return {"mode": self.mode, "solver": "Newton XPBD coarse beads; NOT calibrated sand/MPM",
                "particle_count": self.count, "radius_m": self.radius, "assumed_packing_fraction": self.packing,
                "assumed_bulk_density_kg_m3": 1600., "contact_friction": .6,
                "nominal_bulk_equivalent_liters": self.nominal_liters,
                "retained_count": int(retained.sum()), "not_retained_count": int((~retained).sum()),
                "retained_fraction": ratio, "retained_bulk_equivalent_liters": self.nominal_liters*ratio,
                "below_surface_suspect_count": int(self.below_surface.sum()),
                "below_surface_note": "May include edge escape followed by travel underneath; not proof of seam tunneling.",
                "settled": settled, "tail_max_speed_m_s": max(self.tail_speeds, default=None),
                "tail_max_rms_speed_m_s": max(self.tail_rms, default=None),
                "max_pair_overlap_m": overlap,
                "sim_seconds": self.sim_time, "substeps": self.substeps, "iterations": 8,
                "soft_contact_relaxation": self.relaxation,
                "peak_sampled_contacts": self.peak_contacts,
                "minimum_retention_fraction": .99,
                "diagnostic_retention_pass": ratio >= .99 and settled and not self.below_surface.any() and overlap < .1*self.radius,
                "capacity_2l_physically_validated": False,
                "limitations": "Static blade only, same provisional orientation as robot demo. No arm reaction, no calibrated material, no actual volume measurement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("dose", "probe"), default="dose")
    parser.add_argument("--radius", type=float, default=.006)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--relaxation", type=float, default=.1)
    parser.add_argument("--require-pass", action="store_true", help="exit nonzero if diagnostic acceptance fails")
    args = parser.parse_args()
    options = vars(args).copy()
    require_pass = options.pop("require_pass")
    scene = ToolParticleScene(**options)
    while scene.sim_time < scene.duration-1e-8:
        scene.step()
    report = scene.report()
    output = PROJECT_ROOT / f"logs/tool_particles/{args.mode}_r{args.radius:g}_s{args.substeps}_relax{args.relaxation:g}_t{args.duration:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), initial=scene.initial_positions,
                        final=scene.state_0.particle_q.numpy(), velocity=scene.state_0.particle_qd.numpy())
    print(json.dumps(report, indent=2))
    print(f"REPORT={output}")
    if require_pass and not report["diagnostic_retention_pass"]:
        raise SystemExit("TOOL_PARTICLE_DIAGNOSTIC_FAILED; report saved")


if __name__ == "__main__":
    main()
