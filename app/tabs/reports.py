"""Reports tab — per-model training history, metrics, plots, skipped IDs."""
from __future__ import annotations

import streamlit as st

from _constants import MODELS, REPORTS_ROOT
from _loaders import (
    load_verdict_metrics, load_stance_metrics,
    load_is_metrics, load_training_history, load_test_records,
)


def _render_model_report(model_key: str) -> None:
    vm = load_verdict_metrics(model_key)
    sm = load_stance_metrics(model_key)
    im = load_is_metrics(model_key)
    th = load_training_history(model_key)

    if vm is None and th is None:
        st.info(f"No report files found for **{model_key}** — run `just run model` to generate them.")
        return

    # ── Top metric cards ─────────────────────────────────────────────────────
    if vm:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Verdict Accuracy", f"{vm['accuracy']:.1%}")
        c2.metric("Macro F1",          f"{vm['macro_f1']:.3f}")
        c3.metric("Weighted F1",       f"{vm['weighted_f1']:.3f}")
        skipped = vm.get("skipped", 0)
        c4.metric("Claims Evaluated",  f"{vm['n_claims']}",
                  delta=f"-{skipped} skipped" if skipped else None,
                  delta_color="inverse" if skipped else "off")

    # ── Training loss curve ───────────────────────────────────────────────────
    if th is not None:
        if isinstance(th, list):
            history, cov = th, None
        else:
            history = th.get("history", [])
            cov     = th.get("data_coverage")

        if history:
            st.markdown("#### Training Loss")
            import pandas as pd
            df_loss = pd.DataFrame(
                [{"Epoch": h["epoch"], "Train": h.get("train_loss"), "Val": h.get("val_loss")}
                 for h in history]
            ).set_index("Epoch")
            st.line_chart(df_loss)

        if cov:
            st.markdown("#### Data Coverage")
            col_t, col_v = st.columns(2)
            with col_t:
                st.markdown("**Train**")
                st.markdown(
                    f"Graphs: **{cov.get('train_graphs', '—')}** / "
                    f"{cov.get('train_total', '—')} total  "
                    f"(skipped {cov.get('train_skipped', 0)})"
                )
            with col_v:
                st.markdown("**Val**")
                st.markdown(
                    f"Graphs: **{cov.get('val_graphs', '—')}** / "
                    f"{cov.get('val_total', '—')} total  "
                    f"(skipped {cov.get('val_skipped', 0)})"
                )

    if not vm:
        st.info("Verdict metrics not found — run evaluation to generate them.")
        return

    # ── Detail tabs ───────────────────────────────────────────────────────────
    t_plots, t_verdict, t_stance, t_is, t_skipped = st.tabs(
        ["Plots", "Verdict", "Stance", "Inf. Strength", "Skipped IDs"]
    )

    with t_plots:
        plots_dir = REPORTS_ROOT / model_key / "eval" / "plots"
        imgs = [
            (plots_dir / "confusion_matrix.png",   "Confusion Matrix"),
            (plots_dir / "class_f1.png",            "Per-Class F1"),
            (plots_dir / "per_source_accuracy.png", "Per-Source Accuracy"),
        ]
        cols = st.columns(len(imgs))
        for col, (img_path, caption) in zip(cols, imgs):
            if img_path.exists():
                col.image(str(img_path), caption=caption, width='stretch')
            else:
                col.caption(f"_{caption} — not generated yet_")

    with t_verdict:
        st.markdown("##### Per-Class Breakdown")
        rows = [
            {"Class": cls, "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "N": m["support"]}
            for cls, m in vm.get("per_class", {}).items()
        ]
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Class"), width='stretch')

        conf = vm.get("confusion")
        if conf:
            st.markdown("##### Confusion Matrix")
            labels = list(vm.get("per_class", {}).keys()) or [str(i) for i in range(len(conf))]
            import pandas as pd  # noqa: F811
            st.dataframe(
                pd.DataFrame(conf, index=labels, columns=labels), width='stretch'
            )

        per_src = vm.get("per_source")
        if per_src:
            st.markdown("##### Per-Source Accuracy")
            import pandas as pd  # noqa: F811
            st.dataframe(
                pd.DataFrame(
                    [{"Source": s, "Accuracy": m["accuracy"], "N": m["support"]}
                     for s, m in per_src.items()]
                ).set_index("Source"),
                width='stretch',
            )

    with t_stance:
        if sm:
            ca, cb, cc = st.columns(3)
            ca.metric("Accuracy", f"{sm['accuracy']:.1%}")
            cb.metric("Macro F1", f"{sm['macro_f1']:.3f}")
            cc.metric("ECE",      f"{sm['ece']:.4f}")
            rows = [
                {"Class": cls, "Precision": m["precision"], "Recall": m["recall"],
                 "F1": m["f1"], "N": m["support"]}
                for cls, m in sm.get("per_class", {}).items()
            ]
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows).set_index("Class"), width='stretch')
        else:
            st.caption("stance_metrics.json not found.")

    with t_is:
        if im:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE",      f"{im['rmse']:.4f}")
            c2.metric("Pearson r", f"{im['pearson_r']:.4f}")
            c3.metric("Pred Mean", f"{im['pred_mean']:.4f}")
            c4.metric("True Mean", f"{im['true_mean']:.4f}")
        else:
            st.caption("is_metrics.json not found.")

    with t_skipped:
        skipped_ids = vm.get("skipped_ids", [])
        skipped_count = vm.get("skipped", 0)

        if skipped_count and not skipped_ids:
            st.warning(
                f"**{skipped_count} claims were skipped** during evaluation but their IDs were "
                "not saved in `verdict_metrics.json`. Re-run evaluation to capture them."
            )
        elif skipped_ids:
            st.caption(f"{len(skipped_ids)} claims skipped during evaluation.")
            # skipped_ids may be string claim IDs or integer row indices
            import pandas as pd
            if skipped_ids and isinstance(skipped_ids[0], int):
                # Cross-reference with test records by row position
                test_records = load_test_records()
                idx_to_rec   = {i: r for i, r in enumerate(test_records)}
                rows_skipped = []
                for raw_id in skipped_ids:
                    rec = idx_to_rec.get(raw_id)
                    rows_skipped.append({
                        "Row Index": raw_id,
                        "Claim ID":  rec.get("id", "—") if rec else "—",
                        "Claim":     (rec.get("claim", "—") or "—")[:100] if rec else "—",
                        "Verdict":   (rec.get("verdict") or {}).get("label", "—") if rec else "—",
                    })
            else:
                # String claim IDs — look up claim text
                test_records = load_test_records()
                id_to_rec    = {r.get("id", ""): r for r in test_records}
                rows_skipped = []
                for cid in skipped_ids:
                    rec = id_to_rec.get(str(cid))
                    rows_skipped.append({
                        "Claim ID": str(cid),
                        "Claim":    (rec.get("claim", "—") or "—")[:100] if rec else "—",
                        "Verdict":  (rec.get("verdict") or {}).get("label", "—") if rec else "—",
                    })
            st.dataframe(
                pd.DataFrame(rows_skipped),
                width='stretch',
                height=min(400, 36 * len(rows_skipped) + 38),
            )
        else:
            st.success("No claims were skipped during evaluation.")


def _render_comparison_reports() -> None:
    """Parse and render all comparison_*.md files as coloured DataFrames."""
    import re
    import pandas as pd

    comp_files = sorted(REPORTS_ROOT.glob("comparison_*.md"))
    if not comp_files:
        st.info("`comparison_*.md` files not found in `out/reports/model/`. Run the pipeline to generate them.")
        return

    # Lower-is-better metrics (positive Δ = bad for these)
    _LOWER_IS_BETTER = {"Stance ECE", "IS RMSE"}

    def _parse_md_table(text: str, section: str) -> list[dict]:
        """Extract rows from the markdown table under a given ## heading."""
        # Find the section
        m = re.search(rf"## {re.escape(section)}\n", text)
        if not m:
            return []
        block = text[m.end():]
        rows = []
        for line in block.splitlines():
            if not line.startswith("|"):
                if rows:
                    break
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):  # separator row
                continue
            rows.append(cells)
        return rows

    # Build selector
    labels = [f.stem.replace("comparison_", "").replace("_vs_", "  vs  ") for f in comp_files]
    sel = st.selectbox("Comparison", labels, key="cmp_select", label_visibility="visible")
    idx = labels.index(sel)
    text = comp_files[idx].read_text(encoding="utf-8")

    # Header info
    m_names = comp_files[idx].stem.replace("comparison_", "").split("_vs_")
    model_a, model_b = (m_names[0], m_names[1]) if len(m_names) == 2 else (m_names[0], "?")
    st.markdown(f"### {model_a}  vs  {model_b}")
    gen_m = re.search(r"\*\*Generated:\*\*\s*(.*)", text)
    if gen_m:
        st.caption(f"Generated: {gen_m.group(1).strip()}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    st.markdown("#### Aggregate Metrics")
    agg_rows = _parse_md_table(text, "Aggregate Metrics")
    if agg_rows and len(agg_rows) >= 2:  # header + data rows
        headers = agg_rows[0]
        data    = agg_rows[1:]
        df_agg  = pd.DataFrame(data, columns=headers)
        if len(df_agg.columns) >= 4:
            delta_col = df_agg.columns[-1]

            def _delta_color(val: str, metric: str) -> str:
                val = val.strip()
                try:
                    v = float(val.replace("+", ""))
                except ValueError:
                    return ""
                lower_better = any(lb in metric for lb in _LOWER_IS_BETTER)
                good = (v < 0) if lower_better else (v > 0)
                return "color: #16a34a; font-weight:600" if good else (
                    "color: #dc2626; font-weight:600" if not good and v != 0 else ""
                )

            def _style_row(row: pd.Series) -> list[str]:
                metric = row.iloc[0]
                styles = [""] * (len(row) - 1)
                styles.append(_delta_color(str(row.iloc[-1]), metric))
                return styles

            st.dataframe(
                df_agg.style.apply(_style_row, axis=1),
                width='stretch', hide_index=True,
            )

    # ── Per-source verdict accuracy ────────────────────────────────────────────
    st.markdown("#### Per-Source Verdict Accuracy")
    src_rows = _parse_md_table(text, "Per-Source Verdict Accuracy")
    if src_rows and len(src_rows) >= 2:
        headers_s = src_rows[0]
        data_s    = src_rows[1:]
        df_src    = pd.DataFrame(data_s, columns=headers_s)

        def _style_src_row(row: pd.Series) -> list[str]:
            styles = [""] * (len(row) - 1)
            try:
                v = float(str(row.iloc[-1]).replace("+", ""))
                c = "color: #16a34a; font-weight:600" if v > 0 else (
                    "color: #dc2626; font-weight:600" if v < 0 else ""
                )
            except ValueError:
                c = ""
            styles.append(c)
            return styles

        st.dataframe(
            df_src.style.apply(_style_src_row, axis=1),
            width='stretch', hide_index=True,
        )

    # Raw markdown fallback
    with st.expander("Raw markdown", expanded=False):
        st.markdown(text)


def render() -> None:
    t_models, t_cmp = st.tabs(["Per-Model Reports", "Comparisons"])

    with t_models:
        model_keys = list(MODELS.keys())
        tabs = st.tabs([f"  {k}  " for k in model_keys])
        for tab, mk in zip(tabs, model_keys):
            with tab:
                _render_model_report(mk)

    with t_cmp:
        st.caption(
            "Head-to-head metric comparisons between model pairs. "
            "Green = improvement for left model, red = regression."
        )
        _render_comparison_reports()
