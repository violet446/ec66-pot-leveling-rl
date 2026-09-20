"""Explicit opt-in to generated real visuals and provisional tool calibration."""

import json
import xml.etree.ElementTree as ET
import numpy as np
import warp as wp
from .scene_builder import PROJECT_ROOT


def real_tool_paths():
    directory = PROJECT_ROOT / "generated_assets/real_tool_demo"
    urdf = directory / "ec66_real_visuals.urdf"
    if not urdf.is_file():
        raise FileNotFoundError("Run scripts/prepare_real_tool.py before selecting --real-tool")
    geometry = json.loads((directory / "geometry.json").read_text(encoding="utf-8"))
    root = ET.parse(urdf).getroot()
    transform = wp.transform_identity()
    for name in ("flange_joint", "scoop_mount"):
        origin = root.find(f"joint[@name='{name}']/origin")
        p = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        transform = wp.transform_multiply(transform, wp.transform(wp.vec3(*p), wp.quat_rpy(*rpy)))
    tcp = wp.transform_point(transform, wp.vec3(*geometry["tcp_scoop_local_m"]))
    return urdf, np.array(tcp, dtype=np.float32)
