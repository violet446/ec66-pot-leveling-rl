"""Prescribed moving scoop + XPBD beads. One-way coupling, no robot or PPO."""
import argparse
import json
import numpy as np
import warp as wp
import newton
import trimesh

from .tool_particles import ToolParticleScene, tool_frame, upper_surface
from .scene_builder import PROJECT_ROOT
from ..surface_metrics import SurfaceGrid, heightmap_from_particles


@wp.func
def blend(u: float):
    v = wp.clamp(u, 0.0, 1.0)
    return v*v*v*(10.0 + v*(-15.0 + 6.0*v))


@wp.func
def blend_rate(u: float):
    v = wp.clamp(u, 0.0, 1.0)
    return 30.0*v*v*(1.0-v)*(1.0-v)


@wp.kernel
def drive_tool(counter: wp.array(dtype=int), dt: float,
               body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector)):
    counter[0] += 1
    t = float(counter[0])*dt
    u = (t-20.0)/4.0
    v = (t-24.0)/3.0
    lift = blend(u)
    angle = 1.9*blend(v)
    # Fixed COM at body origin: qd is linear followed by angular velocity.
    body_q[0] = wp.transform(wp.vec3(.2*lift, 0.0, .5+.12*lift),
                             wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle))
    body_qd[0] = wp.spatial_vector(wp.vec3(.2*blend_rate(u)/4.0, 0.0, .12*blend_rate(u)/4.0),
                                    wp.vec3(0.0, 1.9*blend_rate(v)/3.0, 0.0))


def pour_phase(t):
    if t < 20-1e-6:
        return "LOAD_SETTLE"
    if t < 24-1e-6:
        return "CARRY"
    if t < 27-1e-6:
        return "TIP"
    if t < 37-1e-6:
        return "LAND_SETTLE"
    return "DONE"


class ToolPourScene(ToolParticleScene):
    def __init__(self, device="cuda:0"):
        self.counter = wp.zeros(1, dtype=int, device=device)
        self.stage_counts = {}
        self.stage_masks = {}
        self.loading_tail_speeds = []
        self.snapshots = {}
        self.final_grid = SurfaceGrid((.2, 0.), .4, 24)
        super().__init__(device=device, duration=37., substeps=32, relaxation=.1)

    def add_tool(self, builder, cfg):
        # Mass is unused by this prescribed body, but fix COM at the pivot so
        # the analytical linear velocity agrees with the solver's convention.
        self.tool_body = builder.add_body(xform=wp.transform((0., 0., .5), wp.quat_identity()),
                                         mass=1., com=wp.vec3(0.), inertia=wp.mat33(.02,0.,0.,0.,.029,0.,0.,0.,.011), lock_inertia=True,
                                         is_kinematic=True, label="prescribed_real_scoop")
        cfg.is_visible = False
        for i, (center, rotation, size) in enumerate(self.patches):
            builder.add_shape_box(body=self.tool_body,
                xform=wp.transform(wp.vec3(*(center-[0,0,.5])), wp.quat_from_matrix(wp.mat33(rotation))),
                hx=size[0]/2, hy=size[1]/2, hz=size[2]/2, cfg=cfg, label=f"blade/{i}")
        cfg.is_visible = True
        data, rotation, translation = tool_frame()
        mesh = trimesh.load(str(PROJECT_ROOT.parent / "assets/铁锹4y向上.STL"), force="mesh", process=False)
        vertices = (np.asarray(mesh.vertices)*data["config"]["stl_scale_to_m"]
                    -data["config"]["mount_center_raw_m"]) @ rotation.T + translation - [0,0,.5]
        visual = newton.Mesh(vertices.astype(np.float32), mesh.faces.reshape(-1).astype(np.int32), compute_inertia=False)
        builder.add_shape_mesh(body=self.tool_body, mesh=visual,
            cfg=newton.ModelBuilder.ShapeConfig(density=0., has_shape_collision=False, has_particle_collision=False),
            color=(.55,.62,.68), label="real_scoop_visual_only")

    def simulate(self):
        dt = self.frame_dt/self.substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            wp.launch(drive_tool, dim=1, inputs=[self.counter, dt, self.state_0.body_q, self.state_0.body_qd], device=self.model.device)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def carried_mask(self):
        # Used only during untranslated orientation phases, before tipping.
        q = self.state_0.particle_q.numpy()
        transform = self.state_0.body_q.numpy()[0]
        local = q - transform[:3] + [0,0,.5]
        height = upper_surface(local[:, :2], self.patches)
        return np.isfinite(height) & (local[:,2] >= height-self.radius) & (local[:,2] > .2)

    def step(self):
        if self.sim_time >= self.duration-1e-8:
            return
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        q, qd = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("Nonfinite particle state")
        if 19.5 < self.sim_time <= 20.+1e-8:
            self.loading_tail_speeds.append(float(np.max(np.linalg.norm(qd, axis=1))))
        if abs(self.sim_time-20.) < 1e-8 and max(self.loading_tail_speeds, default=float("inf")) >= .05:
            raise RuntimeError("Load has not settled: refusing to start carry")
        for t, name in ((20., "loaded"), (24., "carried_before_tip")):
            if self.sim_time >= t-1e-8 and name not in self.stage_counts:
                self.stage_masks[name] = self.carried_mask()
                self.stage_counts[name] = int(self.stage_masks[name].sum())
        if self.sim_time > self.duration-.5:
            self.tail_speeds.append(float(np.max(np.linalg.norm(qd, axis=1))))
        for t in (20, 24, 27, 37):
            if self.sim_time >= t-1e-8 and t not in self.snapshots:
                self.snapshots[t] = (q.copy(), self.state_0.body_q.numpy()[0].copy())
        self.peak_contacts = max(self.peak_contacts, int(self.contacts.soft_contact_count.numpy()[0]))
        if self.peak_contacts > self.pipeline.soft_contact_max:
            raise RuntimeError("Contact buffer overflow")

    def report(self):
        q = self.state_0.particle_q.numpy()
        # Three disjoint outcomes; tray top is z=0. Upper cutoff admits piles.
        on_tray = (np.abs(q[:,:2]) < 1.-self.radius).all(axis=1) & (q[:,2] < .15) & (q[:,2] > -self.radius)
        in_target = on_tray & (np.linalg.norm(q[:,:2]-[.2,0.], axis=1) < .4)
        outside_target = on_tray & ~in_target
        other = ~on_tray
        height = heightmap_from_particles(q[in_target], self.final_grid, self.radius)
        coverage = float(np.isfinite(height[self.final_grid.mask]).mean())
        settled = bool(self.tail_speeds and max(self.tail_speeds) < .05)
        carry = self.stage_counts.get("carried_before_tip", 0)
        loaded = self.stage_counts.get("loaded", 0)
        loaded_mask = self.stage_masks.get("loaded", np.zeros(self.count, dtype=bool))
        loaded_in_target = int((loaded_mask & in_target).sum())
        centroid = q[in_target,:2].mean(axis=0).tolist() if in_target.any() else None
        return {"task": "prescribed tool carry and pour; NOT EC66 policy", "sim_seconds": self.sim_time,
                "particle_count": self.count, "nominal_bulk_equivalent_liters": self.nominal_liters,
                "stages": self.stage_counts, "carry_retention_fraction": carry/max(loaded,1),
                "preload_not_retained_count": self.count-loaded,
                "loading_tail_max_speed_m_s": max(self.loading_tail_speeds, default=None),
                "loaded_particles_finally_in_target": loaded_in_target,
                "target_landed_count": int(in_target.sum()), "tray_outside_target_count": int(outside_target.sum()),
                "unlanded_or_off_tray_count": int(other.sum()), "target_centroid_xy_m": centroid,
                "target_landed_equivalent_liters": self.nominal_liters*float(in_target.mean()),
                "settled": settled, "tail_max_speed_m_s": max(self.tail_speeds, default=None),
                "heightmap_observed_coverage": coverage, "flatness_reward": None,
                "heightmap_note": "Unobserved cells remain NaN; sparse landing footprint is not a complete pot surface.",
                "substeps": self.substeps, "soft_contact_relaxation": self.relaxation,
                "diagnostic_pass": bool(self.sim_time >= self.duration-1e-8 and loaded >= .99*self.count
                    and carry >= .98*loaded and loaded_in_target >= .95*loaded and settled),
                "limitations": "One-way kinematic blade, no arm or reaction loads; empty tray, no scooping from pot, no PPO; assumed packing 0.6 and uncalibrated material."}

    def save(self):
        report = self.report()
        directory = PROJECT_ROOT / "logs/tool_pour"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        q = self.state_0.particle_q.numpy()
        on_tray = (q[:,2] < .15) & (q[:,2] > -self.radius)
        np.savez_compressed(directory / "landing.npz", positions=q, velocity=self.state_0.particle_qd.numpy(),
                            heights=heightmap_from_particles(q[on_tray], self.final_grid, self.radius))
        np.savez_compressed(directory / "stages.npz",
                            **{f"particles_{t}": value[0] for t,value in self.snapshots.items()},
                            **{f"tool_{t}": value[1] for t,value in self.snapshots.items()})
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scene = ToolPourScene(args.device)
    last = None
    while scene.sim_time < scene.duration-1e-8:
        scene.step()
        phase = pour_phase(scene.sim_time)
        if phase != last:
            print(f"PHASE={phase} t={scene.sim_time:.2f}", flush=True)
            last = phase
    report = scene.save()
    print(json.dumps(report, indent=2))
    if not report["diagnostic_pass"]:
        raise SystemExit("TOOL_POUR_DIAGNOSTIC_FAILED; report saved")


if __name__ == "__main__":
    main()
