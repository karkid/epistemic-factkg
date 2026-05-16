#!/usr/bin/env python3
"""Train EpistemicHGNN (multi-head neuro-symbolic) on the graph dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.epistemic.registry import load_source_trust_registry
from src.model.config import GraphConfig
from src.model.data.featurizer import Featurizer
from src.model.data.builder import ClaimGraphBuilder
from src.model.models import MODELS
from src.model.training.config import TrainConfig
from src.model.training.trainer import Trainer
from src.model.data.types import NUM_STANCE, NUM_VERDICT


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train EpistemicHGNN on the filtered training JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--jsonl", required=True, help="Filtered training JSONL")
    ap.add_argument("--model", default="v1-hgnn", help="Model key from MODELS registry")
    ap.add_argument(
        "--model-name", default="v1-hgnn", help="Display name for logs/reports"
    )
    ap.add_argument("--splits-dir", default="out/data/splits")
    ap.add_argument("--checkpoint-dir", default="out/model/v1-hgnn/checkpoints")
    ap.add_argument("--report-dir", default="out/reports/model/v1-hgnn")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--embed-cache", default="out/model/graphs/embed_cache.pkl")
    ap.add_argument(
        "--dataset", default=None, help="Unused — kept for Justfile compatibility"
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument(
        "--is-loss-weight", type=float, default=0.5, help="λ₁ for IS regression loss"
    )
    ap.add_argument(
        "--verdict-loss-weight",
        type=float,
        default=1.0,
        help="λ₂ for claim-level verdict loss (VerdictHead calibration)",
    )
    ap.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights for both stance and verdict",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap


def _load_split_indices(splits_dir: Path, split: str) -> list[int]:
    path = splits_dir / f"{split}_indices.json"
    if not path.exists():
        print(f"Error: {path} not found. Run 'just split' first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())["indices"]


def _build_graphs(
    records: list[dict],
    indices: list[int],
    builder: ClaimGraphBuilder,
    split_name: str,
    verbose: bool,
) -> list:
    graphs = []
    skipped = 0
    subset = [records[i] for i in indices if i < len(records)]
    for rec in subset:
        try:
            g = builder.build(rec)
            graphs.append(g.data)
        except Exception:
            skipped += 1
    if verbose:
        print(f"  {split_name}: {len(graphs)} graphs  ({skipped} skipped)")
    return graphs


def main() -> None:
    args = _build_parser().parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(
            f"Error: {jsonl_path} not found. Run 'just filter' first.", file=sys.stderr
        )
        sys.exit(1)

    splits_dir = Path(args.splits_dir)
    train_indices = _load_split_indices(splits_dir, "train")
    val_indices = _load_split_indices(splits_dir, "val")

    # ── Load records + build graphs ───────────────────────────────────────────
    if args.verbose:
        print("Loading JSONL records...")
    records = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]

    if args.verbose:
        print("Building graphs...")
    featurizer = Featurizer(cache_path=args.embed_cache)
    registry = load_source_trust_registry(args.registry)
    builder = ClaimGraphBuilder(registry, featurizer)

    train_graphs = _build_graphs(records, train_indices, builder, "train", args.verbose)
    val_graphs = _build_graphs(records, val_indices, builder, "val", args.verbose)
    featurizer.save_cache()

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # ── Class weights (inverse-frequency) ────────────────────────────────────
    stance_weights = None
    verdict_weights = None
    if not args.no_class_weights:
        all_stance_y = torch.cat([g["evidence"].stance_y for g in train_graphs])
        s_counts = (
            torch.bincount(all_stance_y, minlength=NUM_STANCE).float().clamp(min=1.0)
        )
        stance_weights = s_counts.sum() / (NUM_STANCE * s_counts)

        all_verdict_y = torch.cat([g["claim"].y for g in train_graphs])
        v_counts = (
            torch.bincount(all_verdict_y, minlength=NUM_VERDICT).float().clamp(min=1.0)
        )
        verdict_weights = v_counts.sum() / (NUM_VERDICT * v_counts)

        if args.verbose:
            print(
                f"Stance  class weights: {[round(w, 3) for w in stance_weights.tolist()]}"
            )
            print(
                f"Verdict class weights: {[round(w, 3) for w in verdict_weights.tolist()]}"
                f"  (supported / refuted / NEI)"
            )

    # ── Model + trainer ───────────────────────────────────────────────────────
    if args.model not in MODELS:
        print(
            f"Unknown model '{args.model}'. Available: {list(MODELS)}", file=sys.stderr
        )
        sys.exit(1)
    model = MODELS[args.model](
        GraphConfig.v1(), args.hidden_dim, args.heads, args.dropout
    )
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        is_loss_weight=args.is_loss_weight,
        verdict_loss_weight=args.verdict_loss_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = Trainer(
        model,
        config,
        stance_class_weights=stance_weights,
        verdict_class_weights=verdict_weights,
    )

    print(
        f"Training {args.model_name}  |  "
        f"train={len(train_graphs)}  val={len(val_graphs)}  "
        f"device={args.device}  epochs={args.epochs}"
    )
    history = trainer.fit(train_loader, val_loader, verbose=True)

    # ── Save history ──────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best model → {ckpt_dir}/best_model.pt")


if __name__ == "__main__":
    main()
