"""Gate 3B: scripted IK/PD approach, descent, lift and retreat in an empty pot.

This is an engineering motion acceptance test, not a learned scooping policy.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
import newton
from newton.sensors import SensorContact

from .pot_scene import PotConfig, DEFAULT_CONFIG, add_pot
from .scene_builder import (EC66_URDF, INITIAL_JOINT_POS, URDF_VELOCITY_LIMITS,
                            DEFAULT_KP, DEFAULT_KD, _build_ec66_template)
from .reach_env import TCP_OFFSET_LINK6


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def rotation_vector(r):
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-6:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        raise ValueError("IK rotation near 180 degrees; choose a closer initial seed")
    return skew * angle / (2 * np.sin(angle))


class ArmKinematics:
    def __init__(self, urdf_path=EC66_URDF, tcp_offset=TCP_OFFSET_LINK6):
        self.tcp_offset = np.asarray(tcp_offset)
        root = ET.parse(urdf_path).getroot()
        self.origins, self.axes, self.lower, self.upper = [], [], [], []
        for i in range(1, 7):
            joint = root.find(f"joint[@name='joint{i}']")
            origin = joint.find("origin")
            transform = np.eye(4)
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            transform[:3, :3] = axis_rotation([0, 0, 1], yaw) @ axis_rotation([0, 1, 0], pitch) @ axis_rotation([1, 0, 0], roll)
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            self.origins.append(transform)
            self.axes.append(np.fromstring(joint.find("axis").get("xyz"), sep=" "))
            self.lower.append(float(joint.find("limit").get("lower")))
            self.upper.append(float(joint.find("limit").get("upper")))

    def pose(self, q):
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, q):
            rot = np.eye(4)
            rot[:3, :3] = axis_rotation(axis, angle)
            transform = transform @ origin @ rot
        return (transform[:3, 3] + transform[:3, :3] @ self.tcp_offset,
                transform[:3, :3])

    def solve(self, xyz, rotation, seed):
        def error(q):
            p, r = self.pose(q)
            return np.r_[p - xyz, 0.3 * rotation_vector(rotation.T @ r)]
        q = np.array(seed, dtype=float)
        for _ in range(300):
            residual = error(q)
            if np.linalg.norm(residual) < 1e-7:
                break
            jac = np.column_stack([(error(q + np.eye(6)[i] * 1e-5) - residual) / 1e-5 for i in range(6)])
            delta = -np.linalg.solve(jac.T @ jac + 1e-5 * np.eye(6), jac.T @ residual)
            delta *= min(1, 0.15 / max(float(abs(delta).max()), 1e-12))
            for scale in (1, 0.5, 0.25, 0.1):
                trial = np.clip(q + scale * delta, self.lower, self.upper)
                if np.linalg.norm(error(trial)) < np.linalg.norm(residual):
                    q = trial
                    break
            else:
                break
        p, r = self.pose(q)
        if np.linalg.norm(p - xyz) > 0.001 or np.linalg.norm(rotation_vector(rotation.T @ r)) > 0.01:
            raise RuntimeError(f"Unreachable motion waypoint {xyz}; residual={error(q)}")
        return q


class PotMotionScene:
    def __init__(self, device="cuda:0", cfg=None, lateral_offset=0.0, *, real_tool=False):
        self.cfg = cfg or PotConfig.load()
        self.lateral_offset = float(lateral_offset)
        self.real_tool = bool(real_tool)
        self.urdf_path = EC66_URDF
        self.tcp_offset = TCP_OFFSET_LINK6
        if real_tool:
            from .real_tool import real_tool_paths
            self.urdf_path, self.tcp_offset = real_tool_paths()
        self.kin = ArmKinematics(self.urdf_path, self.tcp_offset)
        builder = _build_ec66_template(kp=DEFAULT_KP, kd=DEFAULT_KD, urdf_path=self.urdf_path)
        pot_ids = add_pot(builder, self.cfg)
        self.model = builder.finalize(device=device)
        self.sensor = SensorContact(self.model, sensing_obj_bodies=list(range(6)), counterpart_shapes=pot_ids)
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=False,
                         use_mujoco_contacts=True, njmax=512, nconmax=256)
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0,
                        requested_attributes=self.model.get_requested_contact_attributes())
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.reference_position, self.reference_rotation = self.kin.pose(INITIAL_JOINT_POS)
        actual, _ = self.pose()
        if np.linalg.norm(actual - self.reference_position) > 1e-5:
            raise AssertionError("Independent URDF FK does not match Newton TCP")
        cx, cy = self.cfg.center_xy_m
        # The center descent drives link2 into the near rim at this demo base
        # placement. A local target 15 cm toward the base preserves elbow height.
        tx = cx - 0.15
        # Descend to 8 cm above the nominal half-fill level; no particles yet.
        z_low = self.cfg.floor_z_m + self.cfg.height_m * self.cfg.initial_fill_fraction + 0.08
        self.waypoints = [
            ("approach", np.array([tx, cy + lateral_offset, self.cfg.rim_z_m + 0.30])),
            ("descend", np.array([tx, cy + lateral_offset, z_low])),
            ("lift", np.array([tx, cy + lateral_offset, self.cfg.rim_z_m + 0.30])),
            ("retreat", self.reference_position.copy()),
        ]
        self.joint_waypoints = [INITIAL_JOINT_POS.astype(float)]
        for _, xyz in self.waypoints:
            self.joint_waypoints.append(self.kin.solve(xyz, self.reference_rotation, self.joint_waypoints[-1]))
        self.segment_seconds = [max(2.0, 1.875 * float(np.max(abs(b - a))) / 0.6)
                                for a, b in zip(self.joint_waypoints[:-1], self.joint_waypoints[1:])]
        self.stage = 0
        self.stage_time = 0.0
        self.sim_time = 0.0
        self.frame_dt = 1 / 60
        self.max_contact_n = 0.0
        self.max_speed_rad_s = 0.0
        self.max_orientation_error_rad = 0.0
        self.minimum_tcp_z_m = float(actual[2])
        self.records = []
        self.done = False
        self.dt = self.frame_dt / 4
        self.tracking_bias = np.zeros(6)
        self.fault_reason = None

    def pose(self):
        p = self.state_0.body_q.numpy()[5]
        x, y, z, w = p[3:].astype(float) / np.linalg.norm(p[3:])
        r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return p[:3] + r @ self.tcp_offset, r

    def step(self):
        if self.fault_reason is not None:
            raise RuntimeError(self.fault_reason)
        if self.done:
            return
        duration = self.segment_seconds[self.stage]
        u = min(1.0, (self.stage_time + self.frame_dt) / duration)
        alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
        a, b = self.joint_waypoints[self.stage:self.stage+2]
        desired = a + alpha * (b - a)
        # Bounded integral correction for static PD gravity droop, not model-based
        # gravity compensation. No integration is performed after a contact stop.
        self.tracking_bias = np.clip(self.tracking_bias + 3.0 * self.frame_dt *
            (desired - self.state_0.joint_q.numpy()), -0.10, 0.10)
        self.control.joint_target_pos.assign(np.clip(desired + self.tracking_bias,
            self.kin.lower, self.kin.upper).astype(np.float32))
        for _ in range(4):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.solver.update_contacts(self.contacts, self.state_0)
            self.sensor.update(self.state_0, self.contacts)
            force = float(np.linalg.norm(self.sensor.force_matrix.numpy(), axis=-1).max())
            self.max_contact_n = max(self.max_contact_n, force)
            q, qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
            p, r = self.pose()
            if not all(np.isfinite(v).all() for v in (q, qd, p, r)):
                raise AssertionError("Non-finite motion state")
            if np.any(q < np.array(self.kin.lower) - 1e-4) or np.any(q > np.array(self.kin.upper) + 1e-4):
                raise AssertionError("Joint position limit violated")
            if np.any(abs(qd) > URDF_VELOCITY_LIMITS):
                raise AssertionError("Joint velocity limit violated")
            if force > 0.01:
                matrix = np.linalg.norm(self.sensor.force_matrix.numpy(), axis=-1)
                body, column = np.unravel_index(matrix.argmax(), matrix.shape)
                shape = self.sensor.counterpart_indices[body][column]
                self.fault_reason = (f"Motion stopped: pot contact {force:.3f} N during "
                    f"{self.waypoints[self.stage][0]}, {self.model.body_label[body]} / "
                    f"{self.model.shape_label[shape]}")
                raise RuntimeError(self.fault_reason)
            self.max_speed_rad_s = max(self.max_speed_rad_s, float(abs(qd).max()))
            self.max_orientation_error_rad = max(self.max_orientation_error_rad,
                float(np.linalg.norm(rotation_vector(self.reference_rotation.T @ r))))
            self.minimum_tcp_z_m = min(self.minimum_tcp_z_m, float(p[2]))
            self.sim_time += self.dt
        self.stage_time += self.frame_dt
        if self.stage_time >= duration + 0.6:
            target = self.waypoints[self.stage][1]
            err = float(np.linalg.norm(p - target))
            if err > 0.03:
                raise AssertionError(f"Waypoint error {err:.4f} m in {self.waypoints[self.stage][0]}")
            self.records.append({"phase": self.waypoints[self.stage][0], "target_m": target.tolist(),
                                 "actual_tcp_m": p.tolist(), "error_m": err})
            self.stage += 1
            self.stage_time = 0.0
            self.done = self.stage == len(self.waypoints)

    def verify(self):
        if not self.done:
            raise AssertionError("Motion cycle incomplete; increase --num-frames")
        if self.minimum_tcp_z_m >= self.cfg.rim_z_m - 0.1:
            raise AssertionError("TCP did not enter the pot")
        return {"config": asdict(self.cfg), "lateral_offset_m": self.lateral_offset,
                "real_tool": self.real_tool, "urdf": str(self.urdf_path), "tcp_link6_m": self.tcp_offset.tolist(),
                "phases": self.records, "max_pot_contact_n": self.max_contact_n,
                "max_joint_speed_rad_s": self.max_speed_rad_s,
                "max_orientation_error_rad": self.max_orientation_error_rad,
                "min_tcp_z_m": self.minimum_tcp_z_m, "duration_s": self.sim_time,
                "controller": "scripted URDF IK + smooth joint targets + PD with bounded integral correction; not PPO",
                "self_collision_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", default=None)
    parser.add_argument("--real-tool", action="store_true")
    args = parser.parse_args()
    reports = []
    for offset in (0.0, 0.05, 0.10):
        scene = PotMotionScene(args.device, PotConfig.load(args.config), offset, real_tool=args.real_tool)
        while not scene.done:
            scene.step()
        reports.append(scene.verify())
        print(f"MOTION_CASE_PASS offset={offset} contact={scene.max_contact_n} speed={scene.max_speed_rad_s:.4f}")
    report = Path(args.report or ("logs/real_tool/pot_motion.json" if args.real_tool else "logs/gate3b/pot_motion.json"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"REPORT={report.resolve()}")
    print("GATE3B_SCRIPTED_MOTION_PASS")


if __name__ == "__main__":
    main()
