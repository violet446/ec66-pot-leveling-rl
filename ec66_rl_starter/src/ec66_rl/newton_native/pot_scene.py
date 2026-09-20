"""Gate 3A: open cylindrical pot, EC66 hold and rigid contact probes.

No particles or trained avoidance policy are implied by this diagnostic scene.
Wall boxes overlap at seams; the interior is not a solid convex cylinder.
"""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
import newton
from newton.sensors import SensorContact

from .scene_builder import _build_ec66_template, DEFAULT_KP, DEFAULT_KD, INITIAL_JOINT_POS

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "pot_demo.json"


@dataclass(frozen=True)
class PotConfig:
    outer_diameter_m: float = 2.0
    wall_thickness_m: float = 0.03
    height_m: float = 0.6
    bottom_thickness_m: float = 0.04
    center_xy_m: tuple[float, float] = (1.35, 0.122)
    floor_z_m: float = -0.6
    wall_segments: int = 64
    scoop_volume_liters: float = 2.0
    initial_fill_fraction: float = 0.5
    note: str = "Demo dimensions except the confirmed 2 m outer diameter."

    def __post_init__(self):
        values = [self.outer_diameter_m, self.wall_thickness_m, self.height_m,
                  self.bottom_thickness_m, self.scoop_volume_liters]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Pot dimensions and dose must be finite and positive")
        if self.inner_radius_m <= 0 or not isinstance(self.wall_segments, int) or self.wall_segments < 16:
            raise ValueError("Invalid wall thickness or segment count")
        if len(self.center_xy_m) != 2 or not all(math.isfinite(v) for v in self.center_xy_m) or not math.isfinite(self.floor_z_m):
            raise ValueError("Invalid pot placement")
        if not 0 < self.initial_fill_fraction < 1:
            raise ValueError("Initial fill fraction must lie between zero and one")

    @property
    def inner_radius_m(self):
        return self.outer_diameter_m / 2 - self.wall_thickness_m

    @property
    def rim_z_m(self):
        return self.floor_z_m + self.height_m

    @classmethod
    def load(cls, path=DEFAULT_CONFIG):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def add_pot(builder, cfg):
    cx, cy = cfg.center_xy_m
    # Choose the outer apothem so the box corners stay at exactly the outer radius.
    half_angle = math.pi / cfg.wall_segments
    outer = cfg.outer_diameter_m / 2
    apothem = outer * math.cos(half_angle)
    inner = apothem - cfg.wall_thickness_m
    radius = (inner + apothem) / 2
    half_width = outer * math.sin(half_angle)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    ids = [builder.add_shape_cylinder(
        body=-1, radius=outer, half_height=cfg.bottom_thickness_m / 2,
        xform=wp.transform((cx, cy, cfg.floor_z_m - cfg.bottom_thickness_m / 2), wp.quat_identity()),
        cfg=shape_cfg, label="pot/floor", color=(0.35, 0.42, 0.48))]
    for i in range(cfg.wall_segments):
        angle = i * 2 * math.pi / cfg.wall_segments
        ids.append(builder.add_shape_box(
            body=-1, hx=cfg.wall_thickness_m / 2, hy=half_width,
            hz=cfg.height_m / 2,
            xform=wp.transform((cx + radius * math.cos(angle), cy + radius * math.sin(angle),
                                cfg.floor_z_m + cfg.height_m / 2),
                               wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle)),
            cfg=shape_cfg, label=f"pot/wall_{i:02d}", color=(0.5, 0.58, 0.63)))
    return ids


class PotContactScene:
    """Two rigid balls test the floor and side wall; robot holds its initial pose."""

    def __init__(self, device="cuda:0", cfg=None):
        self.cfg = cfg or PotConfig.load()
        cfg = self.cfg
        builder = _build_ec66_template(kp=DEFAULT_KP, kd=DEFAULT_KD)
        pot_ids = add_pot(builder, cfg)
        cx, cy = cfg.center_xy_m
        self.probe_radius = 0.035
        positions = [(cx, cy, cfg.rim_z_m + 0.15),
                     (cx + cfg.inner_radius_m - 0.15, cy, cfg.floor_z_m + cfg.height_m / 2)]
        self.probe_bodies = []
        for i, pos in enumerate(positions):
            body = builder.add_body(xform=wp.transform(pos, wp.quat_identity()), label=f"probe/{i}")
            builder.add_shape_sphere(body=body, radius=self.probe_radius,
                                    cfg=newton.ModelBuilder.ShapeConfig(density=1000, mu=0.4),
                                    color=(0.9, 0.3 + i * 0.4, 0.1), label=f"probe/ball_{i}")
            self.probe_bodies.append(body)
        self.model = builder.finalize(device=device)
        self.probe_sensor = SensorContact(self.model, sensing_obj_bodies=self.probe_bodies,
                                         counterpart_shapes=pot_ids)
        self.robot_sensor = SensorContact(self.model, sensing_obj_bodies=list(range(6)),
                                         counterpart_shapes=pot_ids)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, disable_contacts=False, use_mujoco_contacts=True,
            njmax=512, nconmax=256)
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0,
                                        requested_attributes=self.model.get_requested_contact_attributes())
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        # Free joints follow the six arm DOFs. Kick only the side-wall probe outward.
        velocity = self.state_0.joint_qd.numpy()
        velocity[12] = 2.0
        self.state_0.joint_qd.assign(velocity)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.dt = 1 / 240
        self.sim_time = 0.0
        self.floor_hit = self.wall_hit = False
        self.max_robot_contact_n = 0.0
        self.max_hold_error_rad = 0.0
        self.min_center_z = positions[0][2]

    def step(self):
        for _ in range(4):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.solver.update_contacts(self.contacts, self.state_0)
            self.probe_sensor.update(self.state_0, self.contacts)
            self.robot_sensor.update(self.state_0, self.contacts)
            force = np.linalg.norm(self.probe_sensor.force_matrix.numpy(), axis=-1)
            self.floor_hit |= bool(force[0, 0] > 1e-4)
            self.wall_hit |= bool(force[1, 1:].max() > 1e-4)
            self.max_robot_contact_n = max(self.max_robot_contact_n,
                float(np.linalg.norm(self.robot_sensor.force_matrix.numpy(), axis=-1).max()))
            q, qd, poses = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy(), self.state_0.body_q.numpy()
            if not all(np.isfinite(a).all() for a in (q, qd, poses)):
                raise AssertionError("Non-finite contact simulation state")
            if np.any(q[:6] < self.model.joint_limit_lower.numpy()[:6] - 1e-4) or np.any(
                    q[:6] > self.model.joint_limit_upper.numpy()[:6] + 1e-4):
                raise AssertionError("Arm joint limit violation")
            self.max_hold_error_rad = max(self.max_hold_error_rad, float(np.max(abs(q[:6] - INITIAL_JOINT_POS))))
            self.min_center_z = min(self.min_center_z, float(poses[self.probe_bodies[0], 2]))
            self.sim_time += self.dt

    def verify(self):
        positions = self.state_0.body_q.numpy()[self.probe_bodies, :3]
        radial = np.linalg.norm(positions[:, :2] - self.cfg.center_xy_m, axis=-1)
        assert self.floor_hit, "Falling probe did not contact the floor"
        assert self.wall_hit, "Moving probe did not contact the side wall"
        assert self.min_center_z < self.cfg.rim_z_m - 0.1, "Interior is obstructed"
        assert np.all(positions[:, 2] >= self.cfg.floor_z_m + self.probe_radius - 0.01), "Floor tunneling"
        assert np.all(radial < self.cfg.inner_radius_m), "Side-wall escape"
        assert self.max_robot_contact_n < 1e-3, "Initial robot placement collides with pot"
        assert self.max_hold_error_rad < 0.06, "Robot hold unstable"
        return {"floor_contact": self.floor_hit, "wall_contact": self.wall_hit,
                "max_robot_pot_force_n": self.max_robot_contact_n,
                "max_hold_error_rad": self.max_hold_error_rad,
                "final_probe_positions_m": positions.tolist(), "sim_time_s": self.sim_time}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", help="optional JSON acceptance report path")
    args = parser.parse_args()
    scene = PotContactScene(args.device, PotConfig.load(args.config))
    for _ in range(180):
        scene.step()
    report = {"config": asdict(scene.cfg), "nominal_scene": scene.verify()}
    # Deliberately intersect the scoop with a raised floor to validate the sensor's
    # positive case. This placement is a diagnostic fault, never the demo scene.
    fault = PotContactScene(args.device, replace(scene.cfg, floor_z_m=0.18))
    for _ in range(3):
        fault.step()
    if fault.max_robot_contact_n <= 0.01:
        raise AssertionError("Robot/pot collision sensor missed deliberate intersection")
    report["deliberate_intersection_detected"] = True
    report["fault_force_n_diagnostic_only"] = fault.max_robot_contact_n
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("GATE3A_POT_CONTACT_PASS")


if __name__ == "__main__":
    main()
