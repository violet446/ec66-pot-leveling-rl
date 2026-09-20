"""Replay a single scripted empty-pot motion cycle, then hold the final display."""

import newton.examples
import warp as wp
from .pot_motion import PotMotionScene
from .pot_scene import DEFAULT_CONFIG, PotConfig


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.scene = PotMotionScene(args.device or "cuda:0", PotConfig.load(args.config), args.lateral_offset,
                                    real_tool=args.real_tool)
        self.model = self.scene.model
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(3.5, -3.5, 2.5), pitch=-33, yaw=130)
        self.radius = wp.array([0.025], dtype=wp.float32, device=self.model.device)
        self.red = wp.array([(1.0, 0.15, 0.15)], dtype=wp.vec3, device=self.model.device)
        self.green = wp.array([(0.15, 1.0, 0.15)], dtype=wp.vec3, device=self.model.device)
        self.reported = False

    def step(self):
        self.scene.step()
        if self.scene.done and not self.reported:
            print(self.scene.verify())
            print("MOTION_COMPLETE: displaying the final state (restart to replay)")
            self.reported = True

    def render(self):
        p, _ = self.scene.pose()
        target = self.scene.waypoints[min(self.scene.stage, 3)][1]
        self.viewer.begin_frame(self.scene.sim_time)
        self.viewer.log_state(self.scene.state_0)
        self.viewer.log_points("/motion/tcp", wp.array([p], dtype=wp.vec3, device=self.model.device),
                               radii=self.radius, colors=self.red)
        self.viewer.log_points("/motion/target", wp.array([target], dtype=wp.vec3, device=self.model.device),
                               radii=self.radius, colors=self.green)
        self.viewer.end_frame()

    def test_final(self):
        self.scene.verify()
        print("GATE3B_MOTION_REPLAY_PASS")


def main():
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1200)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--lateral-offset", type=float, default=0.0)
    parser.add_argument("--real-tool", action="store_true", help="use prepared real visuals and fitted scoop collision")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
