"""Path resolution for config files, so commands work from any directory.

A path in a config is resolved as:

* absolute                -> used as given
* contains ``${VAR}``     -> expanded from the environment, then resolved again
* relative                -> resolved against the repository root, never the
                             current working directory

The research code resolved relative paths against the process's working
directory, which forced training to be launched from the repository root and
evaluation from inside ``evaluation/`` -- two opposite conventions for the same
files.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from muviseg.paths import REPO_ROOT

# Keys whose values are filesystem paths, grouped by config section.
PATH_KEYS: dict[str, tuple[str, ...]] = {
    "DATASET": ("DATA_ROOT", "INSTANCE_MASK_ROOT", "METADATA_PATH", "SEGDATA_ROOT",
                "PAIRS_ROOT", "PAIR_DSC_ROOT"),
    "EVAL": ("PAIRS_FILE", "OUTPUT_DIR"),
    "MODEL": ("CHECKPOINT", "MAST3R_CKPT", "VGGT_CKPT"),
}

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class UnsetVariable(KeyError):
    """A config path referenced an environment variable that is not set."""


def resolve_path(value: str) -> str:
    """Resolve one config path value. Returns a string, as configs hold strings."""
    def sub(m: re.Match) -> str:
        name = m.group(1)
        try:
            return os.environ[name]
        except KeyError:
            raise UnsetVariable(
                f"config path references ${{{name}}}, which is not set in the "
                f"environment; see docs/installation.md"
            ) from None

    expanded = _VAR.sub(sub, value)
    p = Path(expanded).expanduser()
    return str(p if p.is_absolute() else (REPO_ROOT / p))


def resolve_config_paths(cfg: dict) -> dict:
    """Resolve every known path key in a loaded config, in place."""
    for section, keys in PATH_KEYS.items():
        block = cfg.get(section)
        if not isinstance(block, dict):
            continue
        for key in keys:
            if isinstance(block.get(key), str):
                block[key] = resolve_path(block[key])
    return cfg
