"""Check the exact reference poses against phase-2 collision geometry."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ec66_scene import add_bin, add_ec66, add_pedestal
from ec66_trajectory import build_reference_trajectory


def main() -> None:
    trajectory = build_reference_trajectory()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    add_ec66(builder, trajectory.joint_q[0].astype(np.float32), 1200.0, 80.0)
    add_pedestal(builder)
    add_bin(builder)
    model = builder.finalize()
    shape_gaps = model.shape_gap.numpy()
    print(f"PHASE2_SHAPE_GAP_RANGE_M=({float(shape_gaps.min()):.4f}, {float(shape_gaps.max()):.4f})")
    state = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    pairs = set()
    contact_frames = []
    for time_seconds, joint_q in zip(trajectory.times, trajectory.joint_q, strict=True):
        state.joint_q.assign(wp.array(joint_q, dtype=wp.float32, device=model.device))
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        if not count:
            continue
        frame = time_seconds * 24.0 + 1.0
        contact_frames.append(frame)
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        for first, second in zip(shape0, shape1, strict=True):
            first_label = model.shape_label[int(first)] if first >= 0 else "world"
            second_label = model.shape_label[int(second)] if second >= 0 else "world"
            pairs.add(tuple(sorted((first_label, second_label))))

    print(f"PHASE2_REFERENCE_CONTACT_SAMPLE_COUNT={len(contact_frames)}")
    print(f"PHASE2_REFERENCE_CONTACT_FRAME_RANGE={contact_frames[:1] + contact_frames[-1:] if contact_frames else []}")
    print(f"PHASE2_REFERENCE_CONTACT_PAIRS={sorted(pairs)}")
    if contact_frames:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
