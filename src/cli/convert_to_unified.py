"""Convert all registered datasets to a single merged unified v3.0 JSONL.

Usage
-----
python -m src.cli.convert_to_unified \\
    --averitec data/raw/averitec/train.json data/raw/averitec/dev.json \\
    --ai2thor  data/raw/ai2thor/claims_all.jsonl \\
    --synthetic data/raw/synthetic/batch_001.jsonl \\
    --output   out/unified/epistemic_factkg.jsonl

To add a new dataset, implement DatasetConverter in src/adapters/<name>/converter.py
and register it in CONVERTERS below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.adapters.ai2thor.converter import AI2ThorConverter
from src.adapters.averitec.converter import AveritecConverter
from src.core.claims.labels import load_source_trust_registry

_DEFAULT_CONFIG = "configs/config.yaml"
_DEFAULT_REGISTRY = "data/registry/source_trust_registry.jsonl"


def _make_converters(registry_path: str | None = None) -> dict:
    registry: dict = {}
    rp = Path(registry_path or _DEFAULT_REGISTRY)
    if rp.exists():
        registry = load_source_trust_registry(rp)
    return {
        "ai2thor": AI2ThorConverter(),
        "averitec": AveritecConverter(registry=registry),
    }


def convert_to_unified(
    dataset: str,
    in_path: str,
    out_path: str,
    split: str | None = None,
    converters: dict | None = None,
) -> int:
    """Convert a source file to unified v3.0 JSONL. Returns record count.

    For 'synthetic' dataset, the input is already v3.0 JSONL — pass-through only.
    """
    if dataset == "synthetic":
        return _passthrough_jsonl(in_path, out_path)
    reg = converters or _make_converters()
    converter = reg.get(dataset)
    if converter is None:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {sorted(reg)}")
    return converter.convert_file(in_path, out_path, split)


def _passthrough_jsonl(in_path: str, out_path: str) -> int:
    """Copy synthetic JSONL records verbatim — already in v3.0 format."""
    import json
    count = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line:
                fout.write(line + "\n")
                count += 1
    return count


def _infer_split(path: Path) -> str | None:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert all datasets to a single merged unified v3.0 JSONL."
    )
    ap.add_argument(
        "--registry",
        default=_DEFAULT_REGISTRY,
        help=f"Source trust registry JSONL (default: {_DEFAULT_REGISTRY}).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for merged JSONL, e.g. out/unified/epistemic_factkg.jsonl",
    )
    ap.add_argument(
        "--averitec",
        nargs="*",
        default=[],
        metavar="FILE",
        help="AVeriTeC JSON files (top-level list).",
    )
    ap.add_argument(
        "--ai2thor",
        nargs="*",
        default=[],
        metavar="FILE",
        help="AI2THOR JSONL files.",
    )
    ap.add_argument(
        "--synthetic",
        nargs="*",
        default=[],
        metavar="FILE",
        help="Synthetic JSONL files (already in v3.0 format — pass-through).",
    )
    ap.add_argument(
        "--intermediate_dir",
        default=None,
        help="Optional: write per-dataset JSONL files here for debugging.",
    )
    args = ap.parse_args()

    converters = _make_converters(args.registry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    intermediate_dir = Path(args.intermediate_dir) if args.intermediate_dir else None
    if intermediate_dir:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    inputs = (
        [("averitec", p) for p in args.averitec]
        + [("ai2thor", p) for p in args.ai2thor]
        + [("synthetic", p) for p in args.synthetic]
    )

    if not inputs:
        print("No inputs provided. Use --averitec / --ai2thor.")
        print(f"Registered datasets: {sorted(converters)}")
        return

    # Convert each source to a temp file, then merge into the single output.
    import tempfile
    import os

    total_records = 0
    tmp_files = []

    for dataset, in_file in inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"{dataset} input not found: {in_path}")

        split = _infer_split(in_path)
        if dataset == "averitec" and split is None:
            split = "train"

        # Write to intermediate dir (for debugging) or a temp file
        if intermediate_dir:
            inter_path = intermediate_dir / f"{dataset}_{in_path.stem}.jsonl"
            n = convert_to_unified(
                dataset,
                str(in_path),
                str(inter_path),
                split=split,
                converters=converters,
            )
            print(
                f"[{dataset}] {in_path.name} -> {inter_path.name}  ({n} records, split={split})"
            )
            tmp_files.append(str(inter_path))
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as tf:
                tmp_path = tf.name
            n = convert_to_unified(
                dataset, str(in_path), tmp_path, split=split, converters=converters
            )
            print(f"[{dataset}] {in_path.name}  ({n} records, split={split})")
            tmp_files.append(tmp_path)

        total_records += n

    # Merge all into single output
    with open(out_path, "w", encoding="utf-8") as fout:
        for tmp in tmp_files:
            with open(tmp, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)

    # Clean up temp files only (not intermediate_dir files)
    if not intermediate_dir:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    print(f"\nMerged {total_records} records -> {out_path}")


if __name__ == "__main__":
    main()
