"""Interactive launcher for newton.examples "mpm_granular" with a custom block size.

You normally start it through run_block.bat (double-click), which sets up the
ASCII temp / cache paths first.

Usage:
    run_block.bat                        asks for edge length (and bottom height)
    run_block.bat 1.5                    block edge = 1.5 m, bottom height default
    run_block.bat 1.5 2.0                edge = 1.5 m, bottom of the block at z = 2.0
    run_block.bat 4.0 --spread           same particle count, wider looser block
    run_block.bat 1.5 --viewer null      extra newton args can follow

How the particle count is decided (Example.emit_particles does this):
    res   = ceil(3 * edge / voxel_size)
    count = (res + 1) ** 3
So:
  * only changing "edge" keeps voxel_size fixed  -> count grows like edge^3
  * changing edge AND voxel_size by the same factor -> count stays the same,
    the block just gets larger and the points sit further apart ("spread").
    --spread does exactly that: voxel_size = edge / 20, i.e. the spacing and
    the particle count of the official default block (2 m, 226,981 points).
"""

import math
import os
import sys
import tempfile

# --- ASCII paths (this machine's profile path has non-ASCII chars) --------
ASCII_TMP = r"C:\warp_tmp"
ASCII_CACHE = r"C:\warp_cache"
NEWTON_ASSETS = r"C:\newton\assets"
for _d in (ASCII_TMP, ASCII_CACHE, NEWTON_ASSETS):
    os.makedirs(_d, exist_ok=True)
os.environ["TMPDIR"] = ASCII_TMP
os.environ["TEMP"] = ASCII_TMP
os.environ["TMP"] = ASCII_TMP
os.environ["WARP_CACHE_PATH"] = ASCII_CACHE
os.environ["NEWTON_CACHE_PATH"] = NEWTON_ASSETS
tempfile.tempdir = ASCII_TMP

# --- defaults that mirror the official example ---------------------------
PARTICLES_PER_CELL = 3  # hard-coded in Example.emit_particles
VOXEL_SIZE = 0.1  # --voxel-size
DEFAULT_EDGE = 2.0  # official default: emit_hi - emit_lo
DEFAULT_BOTTOM = 1.5  # official default: emit_lo[2]
SPREAD_RATIO = DEFAULT_EDGE / VOXEL_SIZE  # edge / voxel_size = 20 for the default block


def ask(prompt, default):
    try:
        answer = input(prompt).strip()
    except EOFError:
        # no console input available (piped run) - just take the default
        print()
        return default
    return answer if answer else default


def to_float(text, fallback):
    try:
        return float(text)
    except ValueError:
        print(f"  '{text}' is not a number -> using {fallback}")
        return fallback


def main():
    args = sys.argv[1:]

    # the first two bare numbers (not starting with "-") are edge length / bottom height
    positional = []
    rest = []
    for item in args:
        if not rest and not item.startswith("-") and len(positional) < 2:
            positional.append(item)
        else:
            rest.append(item)

    spread = "--spread" in rest
    if spread:
        rest = [item for item in rest if item != "--spread"]  # newton does not know this flag

    explicit_voxel = None
    for index, item in enumerate(rest):
        if item == "--voxel-size" and index + 1 < len(rest):
            explicit_voxel = to_float(rest[index + 1], VOXEL_SIZE)
        elif item.startswith("--voxel-size="):
            explicit_voxel = to_float(item.split("=", 1)[1], VOXEL_SIZE)

    if positional:
        edge = to_float(positional[0], DEFAULT_EDGE)
        bottom = to_float(positional[1], DEFAULT_BOTTOM) if len(positional) > 1 else DEFAULT_BOTTOM
    else:
        # no size given (e.g. the bat was double-clicked): ask interactively
        edge = to_float(
            ask(f"Cube edge length in meters (x, y and z are all this) [{DEFAULT_EDGE}]: ", str(DEFAULT_EDGE)),
            DEFAULT_EDGE,
        )
        bottom = to_float(
            ask(
                f"Start height = z of its bottom face (this is NOT the size, only how high it starts) [{DEFAULT_BOTTOM}]: ",
                str(DEFAULT_BOTTOM),
            ),
            DEFAULT_BOTTOM,
        )
        keep = ask("Keep the particle count the same (wider, looser spacing)? [y/N]: ", "n")
        spread = spread or keep.strip().lower() in ("y", "yes", "1", "true")

    if edge <= 0.0:
        print(f"  edge must be > 0 -> using {DEFAULT_EDGE}")
        edge = DEFAULT_EDGE

    if explicit_voxel is not None and explicit_voxel > 0.0:
        voxel = explicit_voxel
        auto_scaled = False
    elif spread:
        # scale the voxel size with the edge so the count stays constant
        voxel = edge / SPREAD_RATIO
        auto_scaled = True
    else:
        voxel = VOXEL_SIZE
        auto_scaled = False

    half = edge / 2.0
    lo = (-half, -half, bottom)
    hi = (half, half, bottom + edge)

    res = [math.ceil(PARTICLES_PER_CELL * (h - l) / voxel) for l, h in zip(lo, hi)]
    dims = [r + 1 for r in res]
    count = math.prod(dims)

    lo_str = [f"{v:g}" for v in lo]
    hi_str = [f"{v:g}" for v in hi]

    print()
    print("=" * 66)
    print(f"  block edge        : {edge:g} m   (centered on x / y, bottom at z={bottom:g})")
    if auto_scaled:
        print(f"  voxel size        : {voxel:g}   (auto = edge / {SPREAD_RATIO:g}, keeps the count)")
    else:
        print(f"  voxel size        : {voxel:g}")
    print(f"  --emit-lo         : {' '.join(lo_str)}")
    print(f"  --emit-hi         : {' '.join(hi_str)}")
    print(f"  particle grid     : {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"  particle count    : {count}   (estimated, matches the real model)")
    print("=" * 66)
    print()

    extra_args = list(rest)
    if auto_scaled:
        extra_args = ["--voxel-size", f"{voxel:g}", *extra_args]

    # newton.examples.main() expects sys.argv = [prog, example_name, *example_args]
    sys.argv = ["run_block.py", "mpm_granular", "--emit-lo", *lo_str, "--emit-hi", *hi_str, *extra_args]
    print("  launching: python -m newton.examples " + " ".join(sys.argv[1:]))
    print()

    import newton.examples as ex

    ex.main()


if __name__ == "__main__":
    main()
