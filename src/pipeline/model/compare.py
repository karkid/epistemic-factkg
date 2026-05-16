"""Generate a side-by-side comparison report for two evaluated models.

Usage:
  python -m src.pipeline.model.compare \\
      --model1 v1-hgnn --dir1 out/reports/model/v1-hgnn/eval \\
      --model2 baseline --dir2 out/reports/model/baseline/eval \\
      --out out/reports/model/comparison_v1-hgnn_vs_baseline.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(eval_dir: Path, filename: str) -> dict:
    path = eval_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text())


def _delta(a: float, b: float, invert: bool = False) -> str:
    """Format delta between two values. invert=True for metrics where lower is better."""
    d = a - b
    if invert:
        d = -d
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two evaluated models side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model1", required=True, help="Name of first model")
    ap.add_argument("--dir1", required=True, help="Eval dir for model1")
    ap.add_argument("--model2", required=True, help="Name of second model")
    ap.add_argument("--dir2", required=True, help="Eval dir for model2")
    ap.add_argument("--out", required=True, help="Output markdown path")
    args = ap.parse_args()

    dir1, dir2 = Path(args.dir1), Path(args.dir2)
    m1, m2 = args.model1, args.model2

    stance1 = _load(dir1, "stance_metrics.json")
    stance2 = _load(dir2, "stance_metrics.json")
    is1 = _load(dir1, "is_metrics.json")
    is2 = _load(dir2, "is_metrics.json")
    verdict1 = _load(dir1, "verdict_metrics.json")
    verdict2 = _load(dir2, "verdict_metrics.json")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Build comparison rows ─────────────────────────────────────────────────
    agg_rows = [
        ("Stance Accuracy", stance1["accuracy"], stance2["accuracy"], False),
        ("Stance Macro F1", stance1["macro_f1"], stance2["macro_f1"], False),
        ("Stance ECE", stance1["ece"], stance2["ece"], True),
        ("IS RMSE", is1["rmse"], is2["rmse"], True),
        ("IS Pearson r", is1["pearson_r"], is2["pearson_r"], False),
        ("Verdict Accuracy", verdict1["accuracy"], verdict2["accuracy"], False),
        ("Verdict Macro F1", verdict1["macro_f1"], verdict2["macro_f1"], False),
        ("Verdict W-F1", verdict1["weighted_f1"], verdict2["weighted_f1"], False),
    ]

    def _agg_table() -> str:
        header = f"| Metric | {m1} | {m2} | Δ ({m1}−{m2}) |\n"
        sep = "|--------|" + "--------|" * 3 + "\n"
        rows = ""
        for label, v1, v2, inv in agg_rows:
            rows += f"| {label} | {_fmt(v1)} | {_fmt(v2)} | {_delta(v1, v2, inv)} |\n"
        return header + sep + rows

    def _source_table() -> str:
        sources = sorted(set(verdict1["per_source"]) | set(verdict2["per_source"]))
        header = f"| Source | {m1} acc | {m2} acc | Δ |\n"
        sep = "|--------|---------|---------|---|\n"
        rows = ""
        for src in sources:
            a1 = verdict1["per_source"].get(src, {}).get("accuracy", float("nan"))
            a2 = verdict2["per_source"].get(src, {}).get("accuracy", float("nan"))
            n1 = verdict1["per_source"].get(src, {}).get("support", 0)
            n2 = verdict2["per_source"].get(src, {}).get("support", 0)
            d = _delta(a1, a2) if not (a1 != a1 or a2 != a2) else "—"
            rows += f"| {src} | {_fmt(a1)} (n={n1}) | {_fmt(a2)} (n={n2}) | {d} |\n"
        return header + sep + rows

    # ── Write markdown ────────────────────────────────────────────────────────
    md = (
        f"# Model Comparison: {m1} vs {m2}\n\n"
        f"**Generated:** {now}\n\n"
        f"---\n\n"
        f"## Aggregate Metrics\n\n"
        f"> Δ = {m1} − {m2}. "
        f"Positive Δ means {m1} is better (except ECE and RMSE where lower is better).\n\n"
        f"{_agg_table()}\n"
        f"## Per-Source Verdict Accuracy\n\n"
        f"{_source_table()}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Comparison report → {out_path}")


if __name__ == "__main__":
    main()
