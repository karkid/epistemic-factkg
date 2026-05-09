import argparse
from pathlib import Path

from src.pipelines.convert_to_unified import CONVERTERS, convert_to_unified


def _infer_split(path: Path) -> str | None:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def _out_name(dataset: str, in_path: Path) -> str:
    return f"{dataset}_{in_path.stem}.jsonl"


def main():
    ap = argparse.ArgumentParser(
        description="Convert any registered dataset to unified v2.0 JSONL."
    )
    ap.add_argument(
        "--output_dir", required=True, help="Output folder, e.g. data/processed/"
    )
    ap.add_argument(
        "--averitec_inputs",
        nargs="*",
        default=[],
        help="AVeriTeC JSON files (top-level list). Example: train.json dev.json",
    )
    ap.add_argument(
        "--ai2thor_inputs",
        nargs="*",
        default=[],
        help="AI2THOR JSONL files. Example: claims_all.jsonl",
    )
    ap.add_argument(
        "--split_mode",
        choices=["infer", "none"],
        default="infer",
        help="How to set split: 'infer' from filename or 'none'.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    inputs = [("averitec", p) for p in args.averitec_inputs] + [
        ("ai2thor", p) for p in args.ai2thor_inputs
    ]

    for dataset, in_file in inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"{dataset} input not found: {in_path}")

        split = _infer_split(in_path) if args.split_mode == "infer" else None
        if dataset == "averitec" and split is None:
            split = "train"

        out_path = out_dir / _out_name(dataset, in_path)
        n = convert_to_unified(dataset, str(in_path), str(out_path), split=split)
        print(
            f"[{dataset}] {in_path.name} -> {out_path.name}  ({n} records, split={split})"
        )
        total += 1

    if total == 0:
        available = sorted(CONVERTERS)
        print("No inputs provided. Use --averitec_inputs / --ai2thor_inputs.")
        print(f"Registered datasets: {available}")
    else:
        print(f"\nDone. Converted {total} file(s) -> {out_dir}")


if __name__ == "__main__":
    main()
