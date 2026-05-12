#!/usr/bin/env python3
"""Train EpistemicHGNN on the graph dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch_geometric.loader import DataLoader

from src.core.gnn.dataset import EpistemicFactDataset
from src.core.gnn.featurizer import Featurizer
from src.core.gnn.model import EpistemicHGNN
from src.core.gnn.train import Trainer, TrainConfig


def _load_indices(splits_dir: Path, split: str) -> list[int] | None:
    path = splits_dir / f"{split}_indices.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["indices"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train EpistemicHGNN on the graph dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, help="Path to graph_dataset.pt")
    ap.add_argument(
        "--jsonl",
        required=True,
        help="Path to filtered training JSONL (for dataset init)",
    )
    ap.add_argument(
        "--splits-dir", default="out/splits", help="Directory with *_indices.json files"
    )
    ap.add_argument("--checkpoint-dir", default="out/checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default="cpu", help="torch device (cpu / cuda / mps)")
    ap.add_argument(
        "--use-modality-learning", action="store_true", help="Pathway B (Phase 5)"
    )
    ap.add_argument("--aux-loss-weight", type=float, default=0.0)
    ap.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable weighted CrossEntropyLoss",
    )
    ap.add_argument(
        "--no-stance-edges",
        action="store_true",
        help="Zero stance back-edges (supports/refutes/absent/no_evidence) — ablation Run A/B",
    )
    ap.add_argument(
        "--no-epistemic",
        action="store_true",
        help="Also zero has_epistemic edge — ablation Run A only",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Checkpoint sub-directory name (e.g. 'no-stance')",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(
            f"Error: dataset not found: {dataset_path}. Run 'just build-graph' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    splits_dir = Path(args.splits_dir)
    train_idx = _load_indices(splits_dir, "train")
    val_idx = _load_indices(splits_dir, "val")
    if train_idx is None or val_idx is None:
        print(
            f"Error: split files not found in {splits_dir}. Run 'just split' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = EpistemicFactDataset(
        jsonl_path=args.jsonl,
        pt_cache=dataset_path,
        featurizer=Featurizer(),
    )

    train_data = [dataset[i] for i in train_idx if i < len(dataset)]
    val_data = [dataset[i] for i in val_idx if i < len(dataset)]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    class_weights = None if args.no_class_weights else dataset.get_class_weights()

    masked: list[str] = []
    if args.no_stance_edges:
        masked = ["supports", "refutes", "absent", "no_evidence"]
    if args.no_epistemic:
        masked.append("has_epistemic")

    ckpt_dir = args.checkpoint_dir
    if args.run_name:
        ckpt_dir = str(Path(args.checkpoint_dir) / args.run_name)

    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        use_modality_learning=args.use_modality_learning,
        aux_loss_weight=args.aux_loss_weight,
        masked_edge_types=masked,
        device=args.device,
        checkpoint_dir=ckpt_dir,
    )

    model = EpistemicHGNN(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_modality_learning=args.use_modality_learning,
    )

    trainer = Trainer(model=model, class_weights=class_weights, config=config)

    print("=" * 60)
    print("Training EpistemicHGNN")
    print(f"Train graphs  : {len(train_data):,}")
    print(f"Val graphs    : {len(val_data):,}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Hidden dim    : {args.hidden_dim}")
    print(f"Device        : {args.device}")
    print(f"Pathway B     : {args.use_modality_learning}")
    print(f"Masked edges  : {masked or 'none'}")
    print(f"Checkpoint dir: {ckpt_dir}")
    if class_weights is not None:
        print(f"Class weights : {class_weights.tolist()}")
    print("=" * 60)

    history = trainer.fit(train_loader, val_loader, verbose=True)

    best_val_acc = max(h["val_acc"] for h in history)
    print("=" * 60)
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/best_model.pt")
    print("=" * 60)

    history_path = Path(args.checkpoint_dir) / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
