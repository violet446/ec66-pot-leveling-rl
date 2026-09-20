"""View the Gate 3A empty pot, EC66 hold and two rigid contact probes."""

import newton.examples
import warp as wp
from .pot_scene import PotContactScene, PotConfig, DEFAULT_CONFIG


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.scene = PotContactScene(args.device or "cuda:0", PotConfig.load(args.config))
        self.model = self.scene.model
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(3.5, -3.5, 2.5), pitch=-33, yaw=130)

    def step(self):
        self.scene.step()

    def render(self):
        self.viewer.begin_frame(self.scene.sim_time)
        self.viewer.log_state(self.scene.state_0)
        self.viewer.end_frame()

    def test_final(self):
        print(self.scene.verify())
        print("GATE3A_POT_VIEWER_PASS")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
