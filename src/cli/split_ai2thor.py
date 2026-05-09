import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Set


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_list_arg(xs: List[str]) -> List[str]:
    """
    Supports:
      --train_floorplans FloorPlan1 FloorPlan2
    or:
      --train_floorplans FloorPlan1,FloorPlan2
    """
    if not xs:
        return []
    if len(xs) == 1 and "," in xs[0]:
        return [x.strip() for x in xs[0].split(",") if x.strip()]
    return [x.strip() for x in xs if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Split AI2-THOR unified JSONL by floorplan (context_id) without leakage."
    )
    ap.add_argument(
        "--input", required=True, help="Input unified AI2-THOR JSONL (all claims)"
    )
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument(
        "--seed", type=int, default=13, help="Seed for deterministic shuffling"
    )

    ap.add_argument(
        "--mode",
        choices=["pct", "counts", "lists"],
        default="pct",
        help="Split mode: pct (train/dev pct), counts (floorplan counts), lists (explicit floorplan lists)",
    )

    # pct mode
    ap.add_argument("--train_pct", type=int, default=80)
    ap.add_argument("--dev_pct", type=int, default=10)

    # counts mode
    ap.add_argument("--n_train_floorplans", type=int, default=None)
    ap.add_argument("--n_dev_floorplans", type=int, default=None)
    ap.add_argument("--n_test_floorplans", type=int, default=None)

    # lists mode
    ap.add_argument("--train_floorplans", nargs="*", default=[])
    ap.add_argument("--dev_floorplans", nargs="*", default=[])
    ap.add_argument("--test_floorplans", nargs="*", default=[])

    args = ap.parse_args()

    records = read_jsonl(args.input)
    floorplans = sorted(
        {
            r.get("provenance", {}).get("context_id")
            for r in records
            if r.get("provenance", {}).get("context_id")
        }
    )
    n = len(floorplans)

    if n == 0:
        raise ValueError("No provenance.context_id found in input.")

    rng = random.Random(args.seed)

    train_fps: Set[str] = set()
    dev_fps: Set[str] = set()
    test_fps: Set[str] = set()

    if args.mode == "lists":
        train_fps = set(parse_list_arg(args.train_floorplans))
        dev_fps = set(parse_list_arg(args.dev_floorplans))
        test_fps = set(parse_list_arg(args.test_floorplans))

        # validation
        all_given = train_fps | dev_fps | test_fps
        missing = [fp for fp in all_given if fp not in floorplans]
        if missing:
            raise ValueError(f"Unknown floorplans in lists: {missing}")
        overlap = (train_fps & dev_fps) | (train_fps & test_fps) | (dev_fps & test_fps)
        if overlap:
            raise ValueError(f"Overlapping floorplans across splits: {sorted(overlap)}")

        # If user didn't specify all floorplans, put remaining into train by default
        remaining = [fp for fp in floorplans if fp not in all_given]
        train_fps |= set(remaining)

    else:
        fps = floorplans[:]
        rng.shuffle(fps)

        if args.mode == "counts":
            if args.n_train_floorplans is None or args.n_dev_floorplans is None:
                raise ValueError(
                    "--n_train_floorplans and --n_dev_floorplans are required in counts mode."
                )
            n_train = args.n_train_floorplans
            n_dev = args.n_dev_floorplans
            n_test = (
                args.n_test_floorplans
                if args.n_test_floorplans is not None
                else (n - n_train - n_dev)
            )

            if n_train < 1 or n_dev < 0 or n_test < 0:
                raise ValueError("Invalid split counts.")
            if n_train + n_dev + n_test != n:
                raise ValueError(
                    f"Counts must sum to total floorplans ({n}). Got {n_train + n_dev + n_test}."
                )
            train_fps = set(fps[:n_train])
            dev_fps = set(fps[n_train : n_train + n_dev])
            test_fps = set(fps[n_train + n_dev :])

        else:  # pct mode
            if args.train_pct + args.dev_pct >= 100:
                raise ValueError("train_pct + dev_pct must be < 100.")
            # initial rounding
            n_train = int(round(n * args.train_pct / 100))
            n_dev = int(round(n * args.dev_pct / 100))

            # guardrails: ensure at least 1 train; and if possible, at least 1 dev and 1 test
            n_train = max(1, n_train)
            if n >= 3:
                n_dev = max(1, n_dev)
                # ensure at least 1 test
                if n_train + n_dev >= n:
                    n_dev = 1
                    n_train = n - 2
            else:
                # with 1-2 floorplans, true test split not possible
                n_dev = min(n_dev, n - n_train)

            train_fps = set(fps[:n_train])
            dev_fps = set(fps[n_train : n_train + n_dev])
            test_fps = set(fps[n_train + n_dev :])

    buckets = {"train": [], "dev": [], "test": []}
    for r in records:
        fp = r["provenance"]["context_id"]
        if fp in train_fps:
            split = "train"
        elif fp in dev_fps:
            split = "dev"
        else:
            split = "test"
        r["provenance"]["split"] = split
        buckets[split].append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "ai2thor_train.jsonl"
    dev_path = out_dir / "ai2thor_dev.jsonl"
    test_path = out_dir / "ai2thor_test.jsonl"
    write_jsonl(train_path, buckets["train"])
    write_jsonl(dev_path, buckets["dev"])
    write_jsonl(test_path, buckets["test"])

    manifest = {
        "generated_utc": now_utc_iso(),
        "input": args.input,
        "mode": args.mode,
        "seed": args.seed,
        "floorplans_total": n,
        "floorplans": {
            "train": sorted(train_fps),
            "dev": sorted(dev_fps),
            "test": sorted(test_fps),
        },
        "claims": {
            "train": len(buckets["train"]),
            "dev": len(buckets["dev"]),
            "test": len(buckets["test"]),
        },
    }
    manifest_path = out_dir / "ai2thor_splits_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Split complete (configurable)")
    print(f"Floorplans total: {n}")
    print(f"Train floorplans: {len(train_fps)} | claims: {len(buckets['train'])}")
    print(f"Dev   floorplans: {len(dev_fps)} | claims: {len(buckets['dev'])}")
    print(f"Test  floorplans: {len(test_fps)} | claims: {len(buckets['test'])}")
    print(f"Manifest: {manifest_path}")
    print(f"Outputs: {train_path}, {dev_path}, {test_path}")


if __name__ == "__main__":
    main()
