import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def sort_dict_by_value_desc(d: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)


def plot_bar(
    dist: Dict[str, int], title: str, out_path: Path, top_k: Optional[int] = None
):
    items = sort_dict_by_value_desc(dist)
    if top_k is not None:
        items = items[:top_k]

    labels = [str(k) for k, _ in items]
    values = [int(v) for _, v in items]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def md_table_from_dist(title: str, dist: Dict[str, int]) -> str:
    items = sort_dict_by_value_desc(dist)
    lines = []
    lines.append(f"### {title}\n")
    lines.append("| Category | Count |\n")
    lines.append("|---|---:|\n")
    for k, v in items:
        lines.append(f"| {k} | {v} |\n")
    lines.append("\n")
    return "".join(lines)


def summarize_one(summary: Dict[str, Any]) -> str:
    file_name = summary.get("file")
    counts = summary.get("counts", {})
    dists = summary.get("distributions", {})

    parts = []
    parts.append(f"## {file_name}\n\n")
    parts.append("**Counts**\n\n")
    parts.append(f"- total_records: {counts.get('total_records', 0)}\n")
    parts.append(f"- schema_valid: {counts.get('schema_valid', 0)}\n")
    parts.append(f"- schema_invalid: {counts.get('schema_invalid', 0)}\n")
    parts.append(f"- warnings_records: {counts.get('warnings_records', 0)}\n\n")

    if dists.get("verdict_label"):
        parts.append(
            md_table_from_dist("Verdict label distribution", dists["verdict_label"])
        )
    if dists.get("epistemic_proof_type"):
        parts.append(
            md_table_from_dist(
                "Epistemic proof type distribution", dists["epistemic_proof_type"]
            )
        )
    if dists.get("context_type"):
        parts.append(
            md_table_from_dist("Context type distribution", dists["context_type"])
        )
    if dists.get("source_type"):
        parts.append(
            md_table_from_dist(
                "Evidence source_type distribution", dists["source_type"]
            )
        )
    if dists.get("answer_type"):
        parts.append(
            md_table_from_dist("Answer type distribution", dists["answer_type"])
        )

    # Errors/warnings top
    schema_errors_top = summary.get("schema_errors_top") or {}
    logic_warnings_top = summary.get("logic_warnings_top") or {}

    parts.append("### Top schema errors\n\n")
    if not schema_errors_top:
        parts.append("- (none)\n\n")
    else:
        for k, v in list(schema_errors_top.items())[:20]:
            parts.append(f"- {v} × {k}\n")
        parts.append("\n")

    parts.append("### Top logic warnings\n\n")
    if not logic_warnings_top:
        parts.append("- (none)\n\n")
    else:
        for k, v in list(logic_warnings_top.items())[:20]:
            parts.append(f"- {v} × {k}\n")
        parts.append("\n")

    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(
        description="Build dataset report (md + plots) from validation summary JSON."
    )
    ap.add_argument(
        "--summary",
        required=True,
        help="Path to validation_summary.json produced by validate_unified_dataset",
    )
    ap.add_argument(
        "--out_dir", required=True, help="Output directory for report (md + plots)"
    )
    ap.add_argument(
        "--title", default="Epistemic FactKG Dataset Report", help="Report title"
    )
    ap.add_argument(
        "--top_k_sources", type=int, default=15, help="Top-K for source_type plot"
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

    # Build plots for each file
    plot_index_lines = []
    for s in summaries:
        file_name = Path(s.get("file", "unknown")).name.replace(".", "_")
        dists = s.get("distributions", {}) or {}

        # verdict_label
        if dists.get("verdict_label"):
            p = plots_dir / f"{file_name}__verdict_label.png"
            plot_bar(dists["verdict_label"], f"{s.get('file')} — verdict_label", p)
            plot_index_lines.append(
                f"- {s.get('file')} verdict_label: plots/{p.name}\n"
            )

        # epistemic_proof_type
        if dists.get("epistemic_proof_type"):
            p = plots_dir / f"{file_name}__epistemic_proof_type.png"
            plot_bar(
                dists["epistemic_proof_type"],
                f"{s.get('file')} — epistemic_proof_type",
                p,
            )
            plot_index_lines.append(
                f"- {s.get('file')} epistemic_proof_type: plots/{p.name}\n"
            )

        # source_type (top-k)
        if dists.get("source_type"):
            p = plots_dir / f"{file_name}__source_type.png"
            plot_bar(
                dists["source_type"],
                f"{s.get('file')} — source_type (top {args.top_k_sources})",
                p,
                top_k=args.top_k_sources,
            )
            plot_index_lines.append(f"- {s.get('file')} source_type: plots/{p.name}\n")

        # answer_type
        if dists.get("answer_type"):
            p = plots_dir / f"{file_name}__answer_type.png"
            plot_bar(dists["answer_type"], f"{s.get('file')} — answer_type", p)
            plot_index_lines.append(f"- {s.get('file')} answer_type: plots/{p.name}\n")

    # Build markdown report
    md_path = out_dir / "dataset_report.md"
    md_parts = []
    md_parts.append(f"# {args.title}\n\n")
    md_parts.append(f"- Generated (UTC): {now_utc_iso()}\n")
    md_parts.append(f"- Source summary: `{args.summary}`\n\n")

    md_parts.append("## Plots\n\n")
    if plot_index_lines:
        md_parts.extend(plot_index_lines)
        md_parts.append("\n")
        md_parts.append(
            "> Note: image links are relative paths. Open `dataset_report.md` from the report folder.\n\n"
        )
    else:
        md_parts.append("- (no plots generated)\n\n")

    md_parts.append("## Detailed summaries\n\n")
    for s in summaries:
        md_parts.append(summarize_one(s))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_parts))

    # Also write a small manifest
    manifest = {
        "generated_utc": now_utc_iso(),
        "input_summary": args.summary,
        "outputs": {
            "report_md": str(md_path),
            "plots_dir": str(plots_dir),
        },
        "n_files": len(summaries),
    }
    with open(out_dir / "report_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Wrote report: {md_path}")
    print(f"✅ Wrote plots:  {plots_dir}")
    print(f"✅ Wrote manifest: {out_dir / 'report_manifest.json'}")


if __name__ == "__main__":
    main()
