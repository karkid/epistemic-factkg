#!/usr/bin/env python3
"""Evaluate EpistemicHGNN on the held-out test set (ADR-017)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.core.gnn.dataset import EpistemicFactDataset
from src.core.gnn.featurizer import Featurizer
from src.core.gnn.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_ece,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_per_group_accuracy,
    compute_weighted_f1,
)
from src.core.gnn.model import EpistemicHGNN
from src.core.gnn.types import NUM_VERDICT, VERDICT_TO_INT


_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


def _load_indices(path: Path) -> list[int]:
    return json.loads(path.read_text())["indices"]


def _mask_batch(batch, masked_relations: list[str], device: torch.device):
    for rel in masked_relations:
        for et in batch.edge_types:
            if et[1] == rel:
                batch[et].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long, device=device
                )
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate EpistemicHGNN on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--dataset", required=True, help="Path to graph_dataset.pt")
    ap.add_argument(
        "--jsonl", required=True, help="Path to filtered training JSONL (for metadata)"
    )
    ap.add_argument("--splits-dir", default="out/splits")
    ap.add_argument("--output", required=True, help="Directory to write result files")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-stance-edges",
        action="store_true",
        help="Zero stance back-edges (must match training flags)",
    )
    ap.add_argument(
        "--no-epistemic",
        action="store_true",
        help="Zero has_epistemic edge (must match training flags)",
    )
    ap.add_argument(
        "--use-modality-learning",
        action="store_true",
        help="Load model with Pathway B head",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build masked edge relations list ─────────────────────────────────────
    masked: list[str] = []
    if args.no_stance_edges:
        masked = ["supports", "refutes", "absent", "no_evidence"]
    if args.no_epistemic:
        masked.append("has_epistemic")

    # ── Load dataset + test indices ───────────────────────────────────────────
    dataset = EpistemicFactDataset(
        jsonl_path=args.jsonl,
        pt_cache=Path(args.dataset),
        featurizer=Featurizer(),
    )
    test_indices = _load_indices(Path(args.splits_dir) / "test_indices.json")
    test_data = [dataset[i] for i in test_indices if i < len(dataset)]
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── Load Pramana / source metadata from JSONL ─────────────────────────────
    records = [
        json.loads(line)
        for line in Path(args.jsonl).read_text().splitlines()
        if line.strip()
    ]
    pramana_labels = [
        records[i]["epistemic"]["pramana_primary"]
        for i in test_indices
        if i < len(records)
    ]
    source_labels = [
        records[i]["provenance"]["dataset"]
        for i in test_indices
        if i < len(records)
    ]

    # ── Load model ────────────────────────────────────────────────────────────
    model = EpistemicHGNN(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_modality_learning=args.use_modality_learning,
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_logits: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch = _mask_batch(batch, masked, device)
            out = model(batch)
            logits = out["verdict"]
            labels = (batch["claim"].y if hasattr(batch["claim"], "y") else batch.y).view(-1)
            all_logits.append(logits.cpu())
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    preds_t = torch.tensor(all_preds, dtype=torch.long)
    labels_t = torch.tensor(all_labels, dtype=torch.long)
    logits_t = torch.cat(all_logits, dim=0)

    # ── Compute metrics ───────────────────────────────────────────────────────
    accuracy = compute_accuracy(preds_t, labels_t)
    macro_f1 = compute_macro_f1(preds_t, labels_t, NUM_VERDICT)
    weighted_f1 = compute_weighted_f1(preds_t, labels_t, NUM_VERDICT)
    ece = compute_ece(logits_t, labels_t)
    per_class_raw = compute_per_class_metrics(preds_t, labels_t, NUM_VERDICT)
    confusion = compute_confusion_matrix(preds_t, labels_t, NUM_VERDICT)
    per_pramana = compute_per_group_accuracy(preds_t, labels_t, pramana_labels)
    per_source = compute_per_group_accuracy(preds_t, labels_t, source_labels)

    # ── Remap per-class keys to verdict names ─────────────────────────────────
    per_class = {_INT_TO_VERDICT[k]: v for k, v in per_class_raw.items()}

    # ── Write output files ────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": ece,
        "per_class": per_class,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "confusion_matrix.json").write_text(json.dumps(confusion, indent=2))
    (output_dir / "per_pramana.json").write_text(json.dumps(per_pramana, indent=2))
    (output_dir / "per_source.json").write_text(json.dumps(per_source, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Evaluation: {Path(args.checkpoint).parent.name or 'full'}")
    print(f"Test graphs : {len(all_labels)}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"ECE         : {ece:.4f}")
    print()
    print("Per-class:")
    for verdict, m in per_class.items():
        print(
            f"  {verdict:<24} P={m['precision']:.3f}  R={m['recall']:.3f}"
            f"  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("Per-Pramana accuracy:")
    for pramana, m in per_pramana.items():
        print(f"  {pramana:<24} acc={m['accuracy']:.3f}  n={m['support']}")
    print()
    print("Per-source accuracy:")
    for src, m in per_source.items():
        print(f"  {src:<16} acc={m['accuracy']:.3f}  n={m['support']}")
    print(f"\nResults written to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
