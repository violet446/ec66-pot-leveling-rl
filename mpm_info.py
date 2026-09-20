"""Dump model / solver statistics of a Newton example (default: mpm_granular).

Usage (no environment setup needed, the ASCII paths are set below):

    python C:\\newton\\mpm_info.py
    python C:\\newton\\mpm_info.py --voxel-size 0.05
    python C:\\newton\\mpm_info.py --emit-hi 1 1 4.5

Why the header block: this machine's user profile path contains non-ASCII
characters (C:\\Users\\<chinese-name>), which breaks the NVRTC/EDG temporary
files, the CUDA precompiled-header (PCH) folder and the Warp kernel cache.
Forcing ASCII paths before Warp initializes fixes all three.
"""

import math
import os
import sys
import tempfile

ASCII_TMP = r"C:\warp_tmp"
ASCII_CACHE = r"C:\warp_cache"
os.makedirs(ASCII_TMP, exist_ok=True)
os.makedirs(ASCII_CACHE, exist_ok=True)
os.environ["TMPDIR"] = ASCII_TMP
os.environ["TEMP"] = ASCII_TMP
os.environ["TMP"] = ASCII_TMP
os.environ["WARP_CACHE_PATH"] = ASCII_CACHE
tempfile.tempdir = ASCII_TMP

import newton  # noqa: E402
import newton.examples as ex  # noqa: E402
from newton.examples.mpm.example_mpm_granular import Example  # noqa: E402


def _is_scalar(value):
    return isinstance(value, (bool, int, float, str))


def _section(title):
    print()
    print("-" * 62)
    print(title)
    print("-" * 62)


def _dump(obj, keys, indent="  "):
    for key in keys:
        value = getattr(obj, key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) > 8:
            value = f"<{type(value).__name__} len={len(value)}>"
        print(f"{indent}{key:<26} = {value}")


def main():
    extra = sys.argv[1:]
    sys.argv = ["mpm_granular", "--viewer", "null", "--num-frames", "1", "--quiet", *extra]

    parser = Example.create_parser()
    viewer, args = ex.init(parser)
    example = Example(viewer, args)

    model = example.model
    solver = example.solver
    state = example.state_0

    _section("model (newton.Model)")
    _dump(
        model,
        [
            "particle_count",
            "body_count",
            "shape_count",
            "joint_count",
            "tri_count",
            "edge_count",
            "tet_count",
            "spring_count",
        ],
    )

    _section("state buffers")
    for key in ("particle_q", "particle_qd", "body_q", "body_qd"):
        array = getattr(state, key, None)
        if array is not None:
            print(f"  {key:<26} shape = {tuple(array.shape)}, dtype = {array.dtype}")

    _section("per-particle attributes (element 0)")
    for key in ("particle_radius", "particle_mass", "particle_inv_mass", "particle_flags"):
        array = getattr(model, key, None)
        if array is None:
            continue
        try:
            print(f"  {key:<26} = {array.numpy()[0]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {key:<26} = <unavailable: {exc}>")

    _section("MPM solver")
    _dump(
        solver,
        [
            "voxel_size",
            "grid_type",
            "solver",
            "transfer_scheme",
            "integration_scheme",
            "max_active_cell_count",
            "grid_padding",
        ],
    )

    config = getattr(solver, "config", None)
    if config is not None:
        _section("MPM solver config")
        try:
            items = config.items() if isinstance(config, dict) else vars(config).items()
            for key, value in sorted(items):
                if _is_scalar(value):
                    print(f"  {key:<26} = {value}")
        except Exception as exc:  # noqa: BLE001
            print(f"  <cannot introspect config: {exc}>")

    _section("how particle_count is built (Example.emit_particles)")
    particles_per_cell = 3  # hard-coded in the example
    lo = list(args.emit_lo)
    hi = list(args.emit_hi)
    res = [math.ceil(particles_per_cell * (h - l) / args.voxel_size) for l, h in zip(lo, hi)]
    dims = [r + 1 for r in res]
    cell = [(h - l) / r for l, h, r in zip(lo, hi, res)]
    radius = max(cell) * 0.5
    mass = math.prod(cell) * args.density
    print(f"  particles_per_cell         = {particles_per_cell}")
    print(f"  emit_lo                    = {tuple(lo)}")
    print(f"  emit_hi                    = {tuple(hi)}")
    print(f"  voxel_size (--voxel-size)  = {args.voxel_size}")
    print(f"  particle_res               = {tuple(res)}")
    print(f"  grid dims (res + 1)        = {tuple(dims)}")
    print(f"  => particle_count          = {math.prod(dims)}")
    print(f"  particle radius            = {radius:.6f}")
    print(f"  particle mass              = {mass:.6f}   (cell_volume * density)")

    _section("collider / scene")
    print(f"  collider                   = {args.collider}")
    print(f"  gravity                    = {tuple(args.gravity)}")

    _section("simulation timing")
    print(f"  fps                        = {args.fps}")
    print(f"  substeps                   = {args.substeps}")
    print(f"  frame_dt                   = {example.frame_dt:.6f} s")
    print(f"  sim_dt                     = {example.sim_dt:.6f} s")

    _section("all CLI options (defaults + your overrides)")
    for key, value in sorted(vars(args).items()):
        if _is_scalar(value) or (isinstance(value, (list, tuple)) and len(value) <= 4):
            print(f"  {key:<26} = {value}")

    try:
        viewer.close()
    except Exception:  # noqa: BLE001
        pass
    print()
    print("done.")


if __name__ == "__main__":
    main()
