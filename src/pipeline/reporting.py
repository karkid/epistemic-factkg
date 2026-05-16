"""Shared reporting utilities for data and model pipelines."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory (and parents) if it doesn't exist."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(data: dict, path: str | Path, *, indent: int = 2) -> None:
    """Write a dict as pretty-printed JSON."""
    Path(path).write_text(
        json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8"
    )


def write_jsonl(records: list[dict], path: str | Path) -> int:
    """Write a list of dicts to JSONL. Returns number of records written."""
    out = Path(path)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
