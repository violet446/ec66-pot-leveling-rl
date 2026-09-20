"""Newton phase 2: EC66 trajectory with robust bin collision boundaries."""

from __future__ import annotations

import numpy as np

import newton.examples

from run_ec66_phase1 import Example as Phase1Example


class Example(Phase1Example):
    include_bin = True
    enable_contacts = True

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.max_contact_count = 0
        self.contact_pairs = set()

    def step(self):
        super().step()
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        self.max_contact_count = max(self.max_contact_count, count)
        if count:
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
            for first, second in zip(shape0, shape1, strict=True):
                first_label = self.model.shape_label[int(first)] if first >= 0 else "world"
                second_label = self.model.shape_label[int(second)] if second >= 0 else "world"
                self.contact_pairs.add(tuple(sorted((first_label, second_label))))

    def test_final(self):
        super().test_final()
        print(f"EC66_PHASE2_MAX_CONTACT_COUNT={self.max_contact_count}")
        print(f"EC66_PHASE2_CONTACT_PAIRS={sorted(self.contact_pairs)}")
        if self.max_contact_count != 0:
            raise AssertionError(
                "Unexpected robot/bin contact occurred during the reference trajectory; "
                f"maximum simultaneous contacts={self.max_contact_count}"
            )

    @staticmethod
    def create_parser():
        parser = Phase1Example.create_parser()
        parser.set_defaults(controller="impedance")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, arguments = newton.examples.init(parser)
    newton.examples.run(Example(viewer, arguments), arguments)
