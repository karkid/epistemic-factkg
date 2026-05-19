"""Convert all registered datasets to a single merged unified v3.0 JSONL.

Usage
-----
python -m src.pipeline.data.convert_to_unified \\
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
from src.adapters.ai2thor.validator import AI2ThorValidator
from src.adapters.averitec.converter import AveritecConverter
from src.adapters.averitec.validator import AveritecValidator
from src.adapters.synthetic.validator import SyntheticDataValidator
from src.epistemic.registry import load_source_trust_registry
from src.epistemic.schema import CLAIM_SCHEMA

_DEFAULT_CONFIG = "configs/config.yaml"
_DEFAULT_REGISTRY = "data/registry/source_trust_registry.jsonl"

_DATASET_VALIDATORS = {
    "ai2thor": AI2ThorValidator(),
    "averitec": AveritecValidator(),
}


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
    """Copy synthetic JSONL records to out_path, validating each against CLAIM_SCHEMA.

    Invalid records are logged and skipped so they do not corrupt downstream steps.
    """
    import json
    from jsonschema import Draft7Validator

    validator = Draft7Validator(CLAIM_SCHEMA)
    count = 0
    skipped = 0

    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [SKIP] synthetic line {i}: JSON parse error: {exc}", flush=True)
                skipped += 1
                continue

            errors = list(validator.iter_errors(record))
            if errors:
                rec_id = record.get("id", f"synthetic-line-{i}")
                for err in errors:
                    field = ".".join(str(p) for p in err.absolute_path) or "(root)"
                    print(f"  [SKIP] {rec_id}: schema error at {field!r}: {err.message}", flush=True)
                skipped += 1
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if skipped:
        print(
            f"  [synthetic] {count} records written, "
            f"{skipped} skipped (schema invalid) — fix the generator or source data.",
            flush=True,
        )
    return count


def _validate_intermediate(
    dataset: str,
    jsonl_path: str,
    registry: dict | None = None,
) -> bool:
    """Run dataset-specific validation on an intermediate JSONL file.

    Returns True if no errors were found, False otherwise.
    Prints a per-record summary with any issues found.
    """
    import json

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if dataset == "synthetic":
        sv = SyntheticDataValidator(registry=registry or {})
        report = sv.validate_batch(records)
        print(f"  [validate:synthetic] {report.summary()}", flush=True)
        return report.passes

    dv = _DATASET_VALIDATORS.get(dataset)
    if dv is None:
        return True

    issue_count = 0
    for rec in records:
        msgs = dv.check(rec)
        if msgs:
            rec_id = rec.get("id", "?")
            for msg in msgs:
                print(f"  [validate:{dataset}] {rec_id}: {msg}", flush=True)
            issue_count += 1

    total = len(records)
    status = "PASS" if issue_count == 0 else "FAIL"
    print(
        f"  [validate:{dataset}] {total} records, {issue_count} with issues — {status}",
        flush=True,
    )
    return issue_count == 0


def _infer_split(path: Path) -> str | None:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def run(args) -> int:
    """Called by the build dispatcher; args must have: registry, output,
    averitec, ai2thor, synthetic, intermediate_dir."""
    import tempfile
    import os
    from src.epistemic.registry import load_source_trust_registry as _load_registry

    registry: dict = {}
    rp = Path(getattr(args, "registry", None) or _DEFAULT_REGISTRY)
    if rp.exists():
        registry = _load_registry(rp)

    converters = _make_converters(getattr(args, "registry", None))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    intermediate_dir = Path(args.intermediate_dir) if args.intermediate_dir else None
    if intermediate_dir:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    inputs = (
        [("averitec", p) for p in (args.averitec or [])]
        + [("ai2thor", p) for p in (args.ai2thor or [])]
        + [("synthetic", p) for p in (args.synthetic or [])]
    )

    if not inputs:
        print("No inputs provided.")
        return 1

    total_records = 0
    validation_failures = 0
    tmp_files = []

    for dataset, in_file in inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"{dataset} input not found: {in_path}")
        split = _infer_split(in_path)
        if dataset == "averitec" and split is None:
            split = "train"

        if intermediate_dir:
            inter_path = intermediate_dir / f"{dataset}_{in_path.stem}.jsonl"
            n = convert_to_unified(dataset, str(in_path), str(inter_path), split=split, converters=converters)
            print(f"[{dataset}] {in_path.name} -> {inter_path.name}  ({n} records, split={split})")
            tmp_files.append(str(inter_path))
            ok = _validate_intermediate(dataset, str(inter_path), registry=registry)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
                tmp_path = tf.name
            n = convert_to_unified(dataset, str(in_path), tmp_path, split=split, converters=converters)
            print(f"[{dataset}] {in_path.name}  ({n} records, split={split})")
            tmp_files.append(tmp_path)
            ok = _validate_intermediate(dataset, tmp_path, registry=registry)

        total_records += n
        if not ok:
            validation_failures += 1

    with open(out_path, "w", encoding="utf-8") as fout:
        for tmp in tmp_files:
            with open(tmp, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)

    if not intermediate_dir:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    print(f"\nMerged {total_records} records -> {out_path}")
    if validation_failures:
        print(
            f"WARNING: {validation_failures} dataset(s) had validation issues — "
            "see [validate:*] lines above.",
            flush=True,
        )
    return 0


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
    import sys
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
