#!/usr/bin/env python3
"""Evaluate EpistemicHGNN V1 — stance, IS regression, and symbolic verdict."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.epistemic.registry import load_source_trust_registry
from src.model.data.builder import ClaimGraphBuilder
from src.model.data.featurizer import Featurizer
from src.model.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_ece,
    compute_macro_f1,
    compute_pearson_r,
    compute_per_class_metrics,
    compute_per_group_accuracy,
    compute_rmse,
    compute_weighted_f1,
)
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN
from src.model.config import GraphConfig
from src.model.data.types import NUM_STANCE, NUM_VERDICT, VERDICT_TO_INT

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}
_INT_TO_STANCE = {0: "supports", 1: "refutes", 2: "neutral"}


def _write_eval_plots(
    stance_metrics: dict,
    verdict_metrics: dict,
    plots_dir: Path,
) -> None:
    """Write PNG eval plots to plots_dir: confusion matrix, per-class F1, per-source accuracy."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Confusion matrix heatmap ──────────────────────────────────────────────
    confusion = verdict_metrics["confusion"]  # list[list[int]]
    labels = [_INT_TO_VERDICT[i] for i in range(len(confusion))]
    matrix = np.array(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Verdict Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
            )
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ── Per-class F1 bar chart ────────────────────────────────────────────────
    stance_pc = stance_metrics["per_class"]
    verdict_pc = verdict_metrics["per_class"]
    all_classes = list(stance_pc.keys()) + list(verdict_pc.keys())
    precisions = [m["precision"] for m in stance_pc.values()] + [
        m["precision"] for m in verdict_pc.values()
    ]
    f1s = [m["f1"] for m in stance_pc.values()] + [m["f1"] for m in verdict_pc.values()]

    x = np.arange(len(all_classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, precisions, width, label="Precision", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="F1", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision & F1 (Stance + Verdict)")
    ax.axvline(x=len(stance_pc) - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(
        len(stance_pc) / 2 - 0.5, 1.02, "Stance", ha="center", fontsize=8, color="gray"
    )
    ax.text(
        len(stance_pc) + len(verdict_pc) / 2 - 0.5,
        1.02,
        "Verdict",
        ha="center",
        fontsize=8,
        color="gray",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "class_f1.png", dpi=120)
    plt.close(fig)

    # ── Per-source accuracy bar chart ─────────────────────────────────────────
    per_source = verdict_metrics["per_source"]
    sources = list(per_source.keys())
    accs = [per_source[s]["accuracy"] for s in sources]
    ns = [per_source[s]["support"] for s in sources]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sources, accs, color=["#4C9BE8", "#E87B4C", "#5CBE6A"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Verdict Accuracy")
    ax.set_title("Per-Source Verdict Accuracy")
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "per_source_accuracy.png", dpi=120)
    plt.close(fig)


def _load_test_records(jsonl_path: Path, splits_dir: Path) -> list[dict]:
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    split_file = splits_dir / "test_indices.json"
    if split_file.exists():
        indices = json.loads(split_file.read_text(encoding="utf-8"))["indices"]
        return [records[i] for i in indices if i < len(records)]
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate EpistemicHGNN (V1) on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--jsonl",
        required=True,
        help="Filtered training JSONL (test records drawn from this)",
    )
    ap.add_argument("--model", default="v1-hgnn", help="Model key from MODELS registry")
    ap.add_argument("--model-name", default="v1-hgnn", help="Display name for reports")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--embed-cache", default="out/model/graphs/embed_cache.pkl")
    ap.add_argument("--splits-dir", default="out/data/splits")
    ap.add_argument(
        "--output",
        required=True,
        help="Directory to write stance/IS/verdict JSON files",
    )
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model not in MODELS:
        print(
            f"Unknown model '{args.model}'. Available: {list(MODELS)}", file=sys.stderr
        )
        sys.exit(1)
    is_nli = MODELS.get(args.model) is NLIHybridHGNN
    graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
    model = MODELS[args.model](
        graph_cfg, args.hidden_dim, args.heads, args.dropout
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    model.to(device).eval()

    # ── Load test records ─────────────────────────────────────────────────────
    test_records = _load_test_records(Path(args.jsonl), Path(args.splits_dir))
    registry = load_source_trust_registry(args.registry)
    featurizer = Featurizer(cache_path=args.embed_cache, device=args.device)
    builder = ClaimGraphBuilder(registry, featurizer, use_nli=is_nli)

    # ── Accumulators ─────────────────────────────────────────────────────────
    stance_preds: list[torch.Tensor] = []
    stance_ys: list[torch.Tensor] = []
    stance_logits_: list[torch.Tensor] = []
    is_preds: list[torch.Tensor] = []
    is_ys: list[torch.Tensor] = []
    verdict_preds: list[int] = []
    verdict_trues: list[int] = []
    source_labels: list[str] = []

    skipped = 0
    with torch.no_grad():
        for record in test_records:
            try:
                g = builder.build(record)
            except Exception:
                skipped += 1
                continue

            if g is None:
                skipped += 1
                continue

            data = g.data.to(device)
            out = model.predict(data)

            stance_preds.append(out["stance_pred"].cpu())
            stance_ys.append(data["evidence"].stance_y.cpu())
            stance_logits_.append(out["stance_logits"].cpu())
            is_preds.append(out["is_pred"].view(-1).cpu())
            is_ys.append(data["evidence"].is_y.view(-1).cpu())

            v_pred = VERDICT_TO_INT.get(out["verdict"], 2)
            v_true = g.label
            verdict_preds.append(v_pred)
            verdict_trues.append(v_true)
            source_labels.append(g.dataset)

    if not verdict_preds:
        print("No test records evaluated — check splits-dir and jsonl path.")
        return

    # ── Concatenate evidence-level tensors ────────────────────────────────────
    sp_t = torch.cat(stance_preds)
    sy_t = torch.cat(stance_ys)
    sl_t = torch.cat(stance_logits_)
    ip_t = torch.cat(is_preds)
    iy_t = torch.cat(is_ys)
    vp_t = torch.tensor(verdict_preds, dtype=torch.long)
    vy_t = torch.tensor(verdict_trues, dtype=torch.long)

    # ── Stance metrics ────────────────────────────────────────────────────────
    stance_per_class_raw = compute_per_class_metrics(sp_t, sy_t, NUM_STANCE)
    stance_metrics = {
        "accuracy": round(compute_accuracy(sp_t, sy_t), 4),
        "macro_f1": round(compute_macro_f1(sp_t, sy_t, NUM_STANCE), 4),
        "ece": compute_ece(sl_t, sy_t),
        "n_evidence": int(sp_t.numel()),
        "per_class": {_INT_TO_STANCE[k]: v for k, v in stance_per_class_raw.items()},
    }

    # ── IS metrics ────────────────────────────────────────────────────────────
    is_metrics = {
        "rmse": compute_rmse(ip_t, iy_t),
        "pearson_r": compute_pearson_r(ip_t, iy_t),
        "n_evidence": int(ip_t.numel()),
        "pred_mean": round(ip_t.mean().item(), 4),
        "true_mean": round(iy_t.mean().item(), 4),
    }

    # ── Verdict metrics ───────────────────────────────────────────────────────
    verdict_per_class_raw = compute_per_class_metrics(vp_t, vy_t, NUM_VERDICT)
    verdict_metrics = {
        "accuracy": round(compute_accuracy(vp_t, vy_t), 4),
        "macro_f1": round(compute_macro_f1(vp_t, vy_t, NUM_VERDICT), 4),
        "weighted_f1": round(compute_weighted_f1(vp_t, vy_t, NUM_VERDICT), 4),
        "n_claims": len(verdict_preds),
        "skipped": skipped,
        "per_class": {_INT_TO_VERDICT[k]: v for k, v in verdict_per_class_raw.items()},
        "confusion": compute_confusion_matrix(vp_t, vy_t, NUM_VERDICT),
        "per_source": compute_per_group_accuracy(vp_t, vy_t, source_labels),
    }

    # ── Write output files ────────────────────────────────────────────────────
    (output_dir / "stance_metrics.json").write_text(
        json.dumps(stance_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "is_metrics.json").write_text(json.dumps(is_metrics, indent=2), encoding="utf-8")
    (output_dir / "verdict_metrics.json").write_text(
        json.dumps(verdict_metrics, indent=2), encoding="utf-8"
    )

    # ── Write eval_summary.md ─────────────────────────────────────────────────
    _now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _pc_rows(per_class: dict) -> str:
        return "\n".join(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
            for cls, m in per_class.items()
        )

    def _confusion_table(confusion: list) -> str:
        labels = [_INT_TO_VERDICT[i] for i in range(len(confusion))]
        header = "| True \\ Pred | " + " | ".join(labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        rows = [
            f"| {labels[i]} | "
            + " | ".join(str(confusion[i][j]) for j in range(len(labels)))
            + " |"
            for i in range(len(labels))
        ]
        return "\n".join([header, sep] + rows)

    def _source_rows(per_source: dict) -> str:
        return "\n".join(
            f"| {src} | {m['accuracy']:.3f} | {m['support']} |"
            for src, m in per_source.items()
        )

    plots_dir = output_dir / "plots"
    _write_eval_plots(stance_metrics, verdict_metrics, plots_dir)

    eval_md = (
        f"# Evaluation Summary\n\n"
        f"**Model:** {args.model_name}  \n"
        f"**Generated:** {_now}\n\n"
        f"---\n\n"
        f"## Stance Classification\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Accuracy | {stance_metrics['accuracy']:.4f} |\n"
        f"| Macro F1 | {stance_metrics['macro_f1']:.4f} |\n"
        f"| ECE | {stance_metrics['ece']:.4f} |\n"
        f"| N Evidence | {stance_metrics['n_evidence']} |\n\n"
        f"### Per-Class Breakdown\n\n"
        f"| Class | Precision | Recall | F1 | N |\n|-------|-----------|--------|----|---|\n"
        f"{_pc_rows(stance_metrics['per_class'])}\n\n"
        f"## Information Score (IS)\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| RMSE | {is_metrics['rmse']:.4f} |\n"
        f"| Pearson r | {is_metrics['pearson_r']:.4f} |\n"
        f"| Pred Mean | {is_metrics['pred_mean']:.4f} |\n"
        f"| True Mean | {is_metrics['true_mean']:.4f} |\n\n"
        f"## Verdict\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Accuracy | {verdict_metrics['accuracy']:.4f} |\n"
        f"| Macro F1 | {verdict_metrics['macro_f1']:.4f} |\n"
        f"| Weighted F1 | {verdict_metrics['weighted_f1']:.4f} |\n"
        f"| N Claims | {verdict_metrics['n_claims']} |\n"
        f"| Skipped | {verdict_metrics['skipped']} |\n\n"
        f"### Per-Class Breakdown\n\n"
        f"| Class | Precision | Recall | F1 | N |\n|-------|-----------|--------|----|---|\n"
        f"{_pc_rows(verdict_metrics['per_class'])}\n\n"
        f"### Confusion Matrix\n\n"
        f"{_confusion_table(verdict_metrics['confusion'])}\n\n"
        f"### Per-Source Verdict Accuracy\n\n"
        f"| Source | Accuracy | N |\n|--------|----------|---|\n"
        f"{_source_rows(verdict_metrics['per_source'])}\n\n"
        f"---\n\n"
        f"## Plots\n\n"
        f"![Confusion Matrix](plots/confusion_matrix.png)\n\n"
        f"![Per-Class F1](plots/class_f1.png)\n\n"
        f"![Per-Source Accuracy](plots/per_source_accuracy.png)\n"
    )
    (output_dir.parent / "eval_summary.md").write_text(eval_md, encoding="utf-8")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(
        f"{args.model_name} — {len(verdict_preds)} claims  ({int(sp_t.numel())} evidence items)"
    )
    print()
    print("H1 Stance")
    print(
        f"  accuracy  {stance_metrics['accuracy']:.4f}   macro_f1  {stance_metrics['macro_f1']:.4f}   ECE  {stance_metrics['ece']:.4f}"
    )
    for stance, m in stance_metrics["per_class"].items():
        print(
            f"  {stance:<8}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("H2 Inference Strength")
    print(
        f"  RMSE  {is_metrics['rmse']:.4f}   Pearson r  {is_metrics['pearson_r']:.4f}"
    )
    print(
        f"  pred_mean={is_metrics['pred_mean']:.3f}  true_mean={is_metrics['true_mean']:.3f}"
    )
    print()
    print("Symbolic Verdict")
    print(
        f"  accuracy  {verdict_metrics['accuracy']:.4f}   macro_f1  {verdict_metrics['macro_f1']:.4f}   weighted_f1  {verdict_metrics['weighted_f1']:.4f}"
    )
    for verdict, m in verdict_metrics["per_class"].items():
        print(
            f"  {verdict:<22}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("Per-source verdict accuracy:")
    for src, m in verdict_metrics["per_source"].items():
        print(f"  {src:<16}  acc={m['accuracy']:.3f}  n={m['support']}")
    print(f"\nResults -> {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
