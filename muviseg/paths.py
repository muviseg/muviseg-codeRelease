"""
Path constants for the training package.

Single source of truth for locating the segmast3r and vggt submodules
and their vendored dependencies.

Usage in any training module::

    from muviseg.paths import setup_segmast3r_path, setup_vggt_path
    setup_segmast3r_path()   # call once before importing from src.* / mast3r_src.*
    setup_vggt_path()        # call once before importing from vggt.*
"""

from pathlib import Path

# Absolute paths — resolved at import time, not relative to cwd.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SEGMAST3R_ROOT: Path = REPO_ROOT / "third_party" / "segmast3r"
VGGT_ROOT: Path = REPO_ROOT / "third_party" / "vggt"


def setup_segmast3r_path() -> None:
    """Add segmast3r and its vendored deps to sys.path (idempotent)."""
    import sys

    for path in [
        SEGMAST3R_ROOT,
        SEGMAST3R_ROOT / "mast3r_src" / "dust3r",
        SEGMAST3R_ROOT / "mast3r_src" / "dust3r" / "croco",
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def setup_vggt_path() -> None:
    """Add third_party/vggt to sys.path (idempotent)."""
    import sys

    if str(VGGT_ROOT) not in sys.path:
        sys.path.insert(0, str(VGGT_ROOT))
