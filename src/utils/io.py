# src/common/io.py

import json
from typing import Any, Dict, Iterable, List


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write a list of Python dicts into JSONL format.
    Each row is stored as one JSON object per line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return list of dicts.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
