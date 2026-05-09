from __future__ import annotations

from pathlib import Path

import yaml


def load_epistemic_config(path: str | Path) -> dict | None:
    """Read the `epistemic:` section from the pipeline config YAML.

    Returns the raw dict (keys: confidence_weights, pramana_priority_order)
    or None if the section is absent or the file does not exist. Callers
    pass this directly to AveritecConverter(epistemic_config=...).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict):
        return None
    return raw.get("epistemic") or None
