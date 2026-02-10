import argparse
import os
from pathlib import Path

from src.pipelines.convert_averitec_json import convert_averitec_file
from src.pipelines.convert_ai2thor_to_unified_auto import convert_ai2thor_file


def infer_split_from_filename(path: Path):
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "val" in name:
        return "dev"
    if "test" in name:
        return "test"
    return None


def out_name(dataset: str, in_path: Path) -> str:
    # <dataset>_<file_stem>.jsonl
    return f"{dataset}_{in_path.stem}.jsonl"


def main():
    ap = argparse.ArgumentParser(
        description="Convert Averitec + AI2-THOR to unified JSONL into an output directory."
    )

    ap.add_argument("--output_dir", required=True, help="Output folder, e.g., data/processed/")

    # Multiple inputs per dataset
    ap.add_argument(
        "--averitec_inputs",
        nargs="*",
        default=[],
        help="One or more Averitec JSON files (top-level list). Example: train.json dev.json"
    )
    ap.add_argument(
        "--ai2thor_inputs",
        nargs="*",
        default=[],
        help="One or more AI2-THOR JSONL files. Example: raw.jsonl"
    )

    # Split handling
    ap.add_argument(
        "--split_mode",
        choices=["infer", "none"],
        default="infer",
        help="How to set split: infer from filename (train/dev/test) or keep as-is/None."
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    # ---- Averitec ----
    for in_file in args.averitec_inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Averitec input not found: {in_path}")

        split = infer_split_from_filename(in_path) if args.split_mode == "infer" else None
        if split is None:
            # Averitec converter needs split_name for stable ids, so default safely:
            split = "train"

        out_path = out_dir / out_name("averitec", in_path)
        convert_averitec_file(infile=str(in_path), outfile=str(out_path), split_name=split)
        print(f"✅ Averitec: {in_path} -> {out_path} (split={split})")
        total += 1

    # ---- AI2-THOR ----
    for in_file in args.ai2thor_inputs:
        in_path = Path(in_file)
        if not in_path.exists():
            raise FileNotFoundError(f"AI2-THOR input not found: {in_path}")

        split = infer_split_from_filename(in_path) if args.split_mode == "infer" else None

        out_path = out_dir / out_name("ai2thor", in_path)
        convert_ai2thor_file(infile=str(in_path), outfile=str(out_path), split=split)
        print(f"✅ AI2-THOR: {in_path} -> {out_path} (split={split})")
        total += 1

    if total == 0:
        print("⚠️ No inputs provided. Use --averitec_inputs and/or --ai2thor_inputs.")
    else:
        print(f"\n🎉 Done. Converted {total} file(s). Outputs in: {out_dir}")


if __name__ == "__main__":
    main()