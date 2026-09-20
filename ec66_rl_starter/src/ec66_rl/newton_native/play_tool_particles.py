"""Replay isolated static-blade bead diagnostic, then freeze final display."""
import json
import newton.examples
import warp as wp
from .tool_particles import ToolParticleScene


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.scene = ToolParticleScene(args.device or "cuda:0", mode=args.mode,
                                       radius=args.radius, duration=args.duration,
                                       substeps=args.substeps, relaxation=args.relaxation)
        viewer.set_model(self.scene.model)
        viewer.show_particles = True
        viewer.set_camera(pos=wp.vec3(.85, -1., 1.1), pitch=-25, yaw=130)
        self.reported = False

    def step(self):
        self.scene.step()
        if self.scene.sim_time >= self.scene.duration-1e-8 and not self.reported:
            print(json.dumps(self.scene.report(), indent=2))
            self.reported = True

    def render(self):
        self.viewer.begin_frame(self.scene.sim_time)
        self.viewer.log_state(self.scene.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.scene.sim_time < self.scene.duration-1e-8:
            raise AssertionError("Replay did not finish")
        # Execution smoke test is separate from capacity/retention acceptance.
        print("TOOL_PARTICLE_REPLAY_EXECUTION_PASS (not a capacity acceptance)")


def main():
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1260)
    parser.add_argument("--mode", choices=("dose", "probe"), default="dose")
    parser.add_argument("--radius", type=float, default=.006)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--relaxation", type=float, default=.1)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
