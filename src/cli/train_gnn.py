#!/usr/bin/env python3
"""Train EpistemicHGNN (multi-head neuro-symbolic) on the graph dataset.

Phase 3 rewrite: implements multi-task stance + IS loss.
Imports from src.core.gnn.model, train, dataset will be wired up in Phase 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train EpistemicHGNN on the graph dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, help="Path to graph_dataset.pt")
    ap.add_argument("--jsonl", required=True, help="Path to filtered training JSONL")
    ap.add_argument("--splits-dir", default="out/splits")
    ap.add_argument("--checkpoint-dir", default="out/checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--is-loss-weight", type=float, default=0.5,
                    help="λ weight for IS regression loss: total = stance_loss + λ * is_loss")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable weighted CrossEntropyLoss for stance head")
    ap.add_argument("--device", default="cpu", help="torch device (cpu / cuda / mps)")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}. Run 'just graph' first.",
              file=sys.stderr)
        sys.exit(1)

    splits_dir = Path(args.splits_dir)
    for split in ("train", "val"):
        if not (splits_dir / f"{split}_indices.json").exists():
            print(f"Error: {split}_indices.json not found in {splits_dir}. Run 'just split' first.",
                  file=sys.stderr)
            sys.exit(1)

    # TODO Phase 3: wire up EpistemicFactDataset, EpistemicHGNN, TrainConfig, Trainer
    raise NotImplementedError("Phase 3: implement multi-task training loop")


if __name__ == "__main__":
    main()
