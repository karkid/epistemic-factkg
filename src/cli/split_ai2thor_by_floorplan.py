import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split AI2-THOR by floorplan with exact floorplan counts.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_pct", type=int, default=80)
    ap.add_argument("--dev_pct", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.train_pct + args.dev_pct >= 100:
        raise ValueError("train_pct + dev_pct must be < 100")

    # Load all records and collect floorplans
    records = list(read_jsonl(args.input))
    floorplans = sorted({r.get("context", {}).get("context_id") for r in records if r.get("context", {}).get("context_id")})

    n = len(floorplans)
    if n < 3:
        raise ValueError(f"Need at least 3 unique floorplans for train/dev/test. Found {n}: {floorplans}")

    rng = random.Random(args.seed)
    rng.shuffle(floorplans)

    # compute counts by floorplans, force at least 1 dev and 1 test
    n_train = max(1, int(round(n * args.train_pct / 100)))
    n_dev = max(1, int(round(n * args.dev_pct / 100)))
    # ensure room for test
    if n_train + n_dev >= n:
        n_dev = 1
        n_train = n - 2

    train_fps = set(floorplans[:n_train])
    dev_fps = set(floorplans[n_train:n_train + n_dev])
    test_fps = set(floorplans[n_train + n_dev:])

    buckets = {"train": [], "dev": [], "test": []}
    for r in records:
        fp = r["context"]["context_id"]
        if fp in train_fps:
            split = "train"
        elif fp in dev_fps:
            split = "dev"
        else:
            split = "test"
        r["context"]["split"] = split
        buckets[split].append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "ai2thor_train.jsonl"
    dev_path = out_dir / "ai2thor_dev.jsonl"
    test_path = out_dir / "ai2thor_test.jsonl"
    write_jsonl(train_path, buckets["train"])
    write_jsonl(dev_path, buckets["dev"])
    write_jsonl(test_path, buckets["test"])

    print("✅ Split complete (exact floorplans)")
    print(f"Floorplans total: {n}")
    print(f"Train floorplans: {len(train_fps)} | claims: {len(buckets['train'])}")
    print(f"Dev   floorplans: {len(dev_fps)} | claims: {len(buckets['dev'])}")
    print(f"Test  floorplans: {len(test_fps)} | claims: {len(buckets['test'])}")
    print("Train fps:", sorted(train_fps))
    print("Dev fps:", sorted(dev_fps))
    print("Test fps:", sorted(test_fps))
    print(f"Outputs: {train_path}, {dev_path}, {test_path}")

if __name__ == "__main__":
    main()
