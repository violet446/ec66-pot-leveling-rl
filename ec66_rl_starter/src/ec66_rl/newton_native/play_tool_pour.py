"""Visible real scoop + grains + stage display; prescribed motion, not PPO."""
import json
import numpy as np
import warp as wp
import newton.examples
from .tool_pour import ToolPourScene, pour_phase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.scene = ToolPourScene(args.device or "cuda:0")
        if not args.show_loading:
            print("Preparing settled load (20 simulated seconds); carry will start when ready.", flush=True)
            while self.scene.sim_time < 20.-1e-8:
                self.scene.step()
        viewer.set_model(self.scene.model)
        viewer.show_particles = True
        viewer.show_collision = False
        viewer.set_camera(pos=wp.vec3(1.4, -1.5, 1.25), pitch=-27, yaw=135)
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        self.target_ring = wp.array(np.column_stack((.2+.4*np.cos(angles), .4*np.sin(angles), np.full(100,.008))),
                                    dtype=wp.vec3, device=self.scene.model.device)
        self.last_phase = None
        self.reported = False

    def step(self):
        self.scene.step()
        phase = pour_phase(self.scene.sim_time)
        if phase != self.last_phase:
            print(f"PHASE={phase} t={self.scene.sim_time:.2f}", flush=True)
            self.last_phase = phase
        if phase == "DONE" and not self.reported:
            print(json.dumps(self.scene.save(), indent=2))
            print("COMPLETE: final display held. Reset/restart to replay.")
            self.reported = True

    def gui(self, ui):
        ui.text("KINEMATIC SCOOP + BEADS (NO ROBOT / PPO)")
        ui.text(f"Stage: {pour_phase(self.scene.sim_time)}")
        ui.text(f"Simulation: {self.scene.sim_time:.1f} / 37 s")
        ui.text("20-24 carry; 24-27 tip; 27-37 settle")
        ui.text("Green ring = landing target, not pot wall")
        if self.reported:
            ui.text("DONE - final state held. Reset to replay.")

    def render(self):
        self.viewer.begin_frame(self.scene.sim_time)
        self.viewer.log_state(self.scene.state_0)
        self.viewer.log_points("/landing_target", self.target_ring, radii=.003, colors=(.1,.85,.25))
        self.viewer.end_frame()

    def test_final(self):
        report = self.scene.report()
        if not report["diagnostic_pass"]:
            raise AssertionError(report)
        print("TOOL_POUR_REPLAY_PASS")


def main():
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=2280)
    parser.add_argument("--show-loading", action="store_true", help="show initial 20 s settling instead of warming up")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
