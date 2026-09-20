"""Process-local runtime paths that are safe for Warp/NVRTC on Windows."""

from __future__ import annotations

import os
from pathlib import Path


def configure_warp_runtime_paths() -> None:
    """Avoid NVRTC failures caused by non-ASCII Windows user profile paths."""
    workspace_root = Path(__file__).resolve().parents[4]
    temp_dir = workspace_root / "warp_tmp"
    cache_dir = workspace_root / "warp_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["WARP_CACHE_PATH"] = str(cache_dir)

