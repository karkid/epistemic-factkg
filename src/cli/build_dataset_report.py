"""Generate a dataset report (markdown + charts) from validate_unified_dataset output."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sort_desc(d: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)


def plot_bar(
    dist: Dict[str, int], title: str, out_path: Path, top_k: Optional[int] = None
):
    items = sort_desc(dist)
    if top_k:
        items = items[:top_k]
    labels = [str(k) if k is not None else "null" for k, _ in items]
    values = [int(v) for _, v in items]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def md_table(title: str, dist: Dict[str, int], total: int = 0) -> str:
    items = sort_desc(dist)
    lines = [f"### {title}\n\n", "| Category | Count | % |\n", "|---|---:|---:|\n"]
    for k, v in items:
        pct = f"{v / total * 100:.1f}" if total else "-"
        lines.append(f"| {k if k is not None else 'null'} | {v:,} | {pct} |\n")
    lines.append("\n")
    return "".join(lines)


def summarize_one(s: Dict[str, Any]) -> str:
    counts = s.get("counts", {})
    dists = s.get("distributions", {})
    total = counts.get("total_records", 0)
    file_name = s.get("file", "unknown")

    parts = [f"## {file_name}\n\n"]

    parts.append("**Record counts**\n\n")
    parts.append(f"- Total records: {total:,}\n")
    parts.append(f"- Schema valid: {counts.get('schema_valid', 0):,}\n")
    parts.append(f"- Schema invalid: {counts.get('schema_invalid', 0):,}\n")
    parts.append(
        f"- Records with logic warnings: {counts.get('logic_warnings_records', 0):,}\n\n"
    )

    if dists.get("verdict_label"):
        parts.append(md_table("Verdict distribution", dists["verdict_label"], total))

    if dists.get("evidence_types_all"):
        parts.append(
            md_table(
                "Evidence type distribution", dists["evidence_types_all"], total
            )
        )

    if dists.get("evidence_stance"):
        parts.append(md_table("Evidence stance distribution", dists["evidence_stance"]))

    if dists.get("evidence_modality"):
        parts.append(
            md_table("Evidence modality distribution", dists["evidence_modality"])
        )

    if dists.get("reasoning_structural"):
        parts.append(
            md_table(
                "Claim structure distribution", dists["reasoning_structural"], total
            )
        )

    if dists.get("dataset"):
        parts.append(md_table("Dataset breakdown", dists["dataset"], total))

    schema_errors_top = s.get("schema_errors_top") or {}
    logic_warnings_top = s.get("logic_warnings_top") or {}

    parts.append("### Top schema errors\n\n")
    if not schema_errors_top:
        parts.append("- (none)\n\n")
    else:
        for k, v in list(schema_errors_top.items())[:10]:
            parts.append(f"- {v}x  {k}\n")
        parts.append("\n")

    parts.append("### Top logic warnings\n\n")
    if not logic_warnings_top:
        parts.append("- (none)\n\n")
    else:
        for k, v in list(logic_warnings_top.items())[:10]:
            parts.append(f"- {v}x  {k}\n")
        parts.append("\n")

    # Dataset-level warnings
    dw = s.get("dataset_warnings") or []
    if dw:
        parts.append("### Dataset-level warnings\n\n")
        for w in dw:
            parts.append(f"- **!** {w}\n")
        parts.append("\n")

    # GNN readiness
    gnn = s.get("gnn_readiness") or {}
    if gnn:
        parts.append("### GNN readiness\n\n")
        parts.append(
            f"- Absence claims (non_apprehension): {gnn.get('absence_claims', 0):,} "
            f"({gnn.get('absence_pct', 0):.1f}%)\n"
        )
        balance_ok = gnn.get("label_balance_ok", None)
        balance_str = "Yes" if balance_ok else "No — imbalanced (>70% one label)"
        parts.append(f"- Label balance OK: {balance_str}\n")
        parts.append("\n")

    return "".join(parts)


def plot_horizontal_bar_with_target(
    dist: Dict[str, int],
    targets: Dict[str, int],
    title: str,
    out_path: Path,
):
    """Horizontal bar chart with actual vs target annotations (for Pramana distribution)."""
    labels = list(dist.keys())
    actual_vals = [dist[k] for k in labels]
    target_vals = [targets.get(k, 0) for k in labels]

    y = range(len(labels))
    plt.figure(figsize=(9, max(4, len(labels) * 0.8)))
    plt.barh(list(y), actual_vals, label="Actual", color="#4c72b0")
    plt.barh(list(y), target_vals, label="ADR-012 Target", color="#dd8452", alpha=0.5)
    plt.yticks(list(y), labels)
    plt.xlabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pie(dist: Dict[str, int], title: str, out_path: Path):
    labels = list(dist.keys())
    values = [dist[k] for k in labels]
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_training(
    tv: Dict[str, Any], plots_dir: Path, plot_refs: List[str]
) -> str:
    """Generate the training dataset section for the report markdown."""
    total = tv.get("total_records", 0)
    et_dist = tv.get("evidence_type_distribution", {})
    et_targets = {
        k: v["target"] for k, v in tv.get("evidence_type_vs_targets", {}).items()
    }
    verdict_dist = tv.get("verdict_distribution", {})
    source_dist = tv.get("source_distribution", {})
    warnings = tv.get("warnings", [])
    passed = tv.get("pass", False)
    postulation_count = tv.get("postulation_derivation_count", 0)

    # Plot 1: Evidence type actual vs target (horizontal bar)
    if et_dist and et_targets:
        p = plots_dir / "training_evidence_type_distribution.png"
        plot_horizontal_bar_with_target(
            et_dist,
            et_targets,
            "Training — Evidence type distribution vs ADR-006 targets",
            p,
        )
        plot_refs.append(f"- [Training: Evidence type vs targets](plots/{p.name})\n")

    # Plot 2: Source split pie
    if source_dist:
        p = plots_dir / "training_source_split.png"
        plot_pie(source_dist, "Training — Source split (AI2THOR vs AVeriTeC)", p)
        plot_refs.append(f"- [Training: Source split](plots/{p.name})\n")

    # Plot 3: Verdict distribution bar
    if verdict_dist:
        p = plots_dir / "training_verdict_distribution.png"
        plot_bar(verdict_dist, "Training — Verdict distribution", p)
        plot_refs.append(f"- [Training: Verdict distribution](plots/{p.name})\n")

    parts = ["## Training Dataset (ADR-005 / ADR-006)\n\n"]
    status = "PASS" if passed else "FAIL"
    parts.append(
        f"**Validation status:** {status}  |  **Total records:** {total:,}  |  `postulation_derivation`: {postulation_count}\n\n"
    )

    # Source split table
    parts.append("### Source split\n\n| Source | Count | % |\n|---|---:|---:|\n")
    for src, count in sorted(source_dist.items()):
        pct = f"{count / total * 100:.1f}" if total else "-"
        parts.append(f"| {src} | {count:,} | {pct} |\n")
    parts.append("\n")

    # Evidence type vs targets
    pv = tv.get("evidence_type_vs_targets", {})
    parts.append("### Evidence type distribution vs ADR-006 targets\n\n")
    parts.append(
        "| Evidence type | Actual | Target | Delta | % |\n|---|---:|---:|---:|---:|\n"
    )
    for et in sorted(pv.keys()):
        info = pv[et]
        delta_str = f"{info['delta']:+d}"
        parts.append(
            f"| {et} | {info['actual']:,} | {info['target']:,} | {delta_str} | {info['pct']} |\n"
        )
    parts.append("\n")

    # Verdict distribution
    if verdict_dist:
        parts.append(md_table("Verdict distribution", verdict_dist, total))

    if warnings:
        parts.append("### Warnings\n\n")
        for w in warnings:
            parts.append(f"- **!** {w}\n")
        parts.append("\n")

    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description="Build dataset report (md + plots) from validation summary JSON."
    )
    ap.add_argument("--summary", required=True, help="Path to validation.json")
    ap.add_argument("--out_dir", required=True, help="Output directory for report")
    ap.add_argument("--title", default="Epistemic FactKG Dataset Report")
    ap.add_argument(
        "--training-summary",
        default=None,
        help="Optional path to training_validation.json (adds training section + plots)",
    )
    args = ap.parse_args()

    src = load_json(args.summary)
    summaries = src.get("summaries", [])
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("Expected {summaries: [...]} in the input JSON.")

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    plot_refs: List[str] = []
    for s in summaries:
        tag = Path(s.get("file", "unknown")).name.replace(".", "_")
        dists = s.get("distributions", {}) or {}

        for dist_key, label in [
            ("verdict_label", "Verdict distribution"),
            ("evidence_types_all", "Evidence type distribution"),
            ("evidence_stance", "Evidence stance distribution"),
            ("evidence_modality", "Evidence modality"),
        ]:
            if dists.get(dist_key):
                p = plots_dir / f"{tag}__{dist_key}.png"
                plot_bar(dists[dist_key], f"{s.get('file')} — {label}", p)
                plot_refs.append(f"- [{label}](plots/{p.name})\n")

    training_md: Optional[str] = None
    if args.training_summary:
        tv_path = Path(args.training_summary)
        if tv_path.exists():
            tv = load_json(str(tv_path))
            training_md = summarize_training(tv, plots_dir, plot_refs)

    md_path = out_dir / "summary.md"
    md = []
    md.append(f"# {args.title}\n\n")
    md.append(f"- Generated (UTC): {now_utc_iso()}\n")
    md.append(f"- Source: `{args.summary}`\n\n")

    if plot_refs:
        md.append("## Charts\n\n")
        md.extend(plot_refs)
        md.append("\n")

    if training_md:
        md.append(training_md)

    md.append("## Detailed summaries\n\n")
    for s in summaries:
        md.append(summarize_one(s))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    manifest = {
        "generated_utc": now_utc_iso(),
        "input_summary": args.summary,
        "report_md": str(md_path),
        "plots_dir": str(plots_dir),
        "n_files": len(summaries),
    }
    with open(out_dir / "report_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote report: {md_path}")
    print(f"Wrote plots:  {plots_dir}")


if __name__ == "__main__":
    main()
