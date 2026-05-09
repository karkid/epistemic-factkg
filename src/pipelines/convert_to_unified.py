"""
Single entry point for converting any registered dataset to unified v2.0 JSONL.

Usage
-----
from src.pipelines.convert_to_unified import convert_to_unified

n = convert_to_unified("ai2thor", "data/raw/ai2thor/claims_all.jsonl",
                        "data/processed/ai2thor_unified.jsonl", split="train")

To add a new dataset, implement DatasetConverter in src/adapters/<name>/converter.py
and register it in CONVERTERS below.
"""
from __future__ import annotations

from src.adapters.ai2thor.converter import AI2ThorConverter
from src.adapters.averitec.converter import AveritecConverter

CONVERTERS = {
    "ai2thor": AI2ThorConverter(),
    "averitec": AveritecConverter(),
}


def convert_to_unified(
    dataset: str,
    in_path: str,
    out_path: str,
    split: str | None = None,
) -> int:
    """
    Convert a source file to unified v2.0 JSONL.

    Parameters
    ----------
    dataset : str
        Dataset name matching a key in CONVERTERS (e.g. 'ai2thor', 'averitec').
    in_path  : str — path to source file
    out_path : str — destination JSONL path
    split    : str | None — 'train' | 'dev' | 'test' | None

    Returns
    -------
    int — number of records written
    """
    converter = CONVERTERS.get(dataset)
    if converter is None:
        available = sorted(CONVERTERS)
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {available}")
    return converter.convert_file(in_path, out_path, split)
