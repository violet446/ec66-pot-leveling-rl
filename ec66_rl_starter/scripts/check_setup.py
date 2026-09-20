"""Report the prerequisites for each route in the learning roadmap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Python: {sys.version.split()[0]}")
    print(f"Starter folder: {root}")
    for module in ("newton", "torch", "rsl_rl", "isaaclab", "onnx", "onnxruntime"):
        print(f"{module:12}: {'READY' if available(module) else 'missing'}")
    if available("newton"):
        import newton

        print(f"Newton version: {getattr(newton, '__version__', 'unknown')}")
    ec66_scene = root.parents[0] / "ec66_sand" / "run_ec66_phase1.py"
    print(f"Existing EC66 scene: {'found' if ec66_scene.exists() else 'not found'} ({ec66_scene})")
    print()
    if all(available(module) for module in ("torch", "rsl_rl", "isaaclab")):
        print("Gate 1 dependencies are ready. Follow docs/roadmap.md to train the official Cartpole task.")
    else:
        print("Gate 0 is ready. Gate 1 still needs PyTorch + RSL-RL + Isaac Lab; do not start EC66 training yet.")


if __name__ == "__main__":
    main()
