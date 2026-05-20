"""Data tab — dataset insights, schema viewer, and claim search/filter."""
from __future__ import annotations

import json

import streamlit as st

from _constants import DATA_REPORT_DIR, UNIFIED_JSONL, GRAPH_CACHE_DIR, INT_TO_VERDICT
from _loaders import load_dataset_stats, load_all_records_list, load_graph_cache, build_graph_id_map
from _state import load_record_into_state


# ── Stats / validation section ────────────────────────────────────────────────

def _render_stats(stats: dict) -> None:
    import pandas as pd

    counts  = stats.get("counts", {})
    cov     = stats.get("coverage", {})
    splits  = stats.get("splits", {})
    dists   = stats.get("distributions", {})
    training = stats.get("training", {})
    schema_errs   = stats.get("schema_errors", {})
    logic_warns   = stats.get("logic_warnings", {})

    # ── Top-line metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",  counts.get("total_records",  "—"))
    c2.metric("Schema Valid",   counts.get("schema_valid",   "—"))
    c3.metric("Schema Invalid", counts.get("schema_invalid", "—"))
    c4.metric("Logic Warnings", counts.get("logic_warnings_records", "—"))
    total_ev = cov.get("evidence_count_sum", "—")
    c5.metric("Total Evidence", total_ev)

    # ── Split sizes ───────────────────────────────────────────────────────────
    if splits:
        st.markdown("#### Split Sizes")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Train", splits.get("train", "—"))
        sc2.metric("Val",   splits.get("val",   "—"))
        sc3.metric("Test",  splits.get("test",  "—"))

    # ── Distributions row ─────────────────────────────────────────────────────
    st.markdown("#### Distributions")
    t_dist, t_types, t_mod, t_stance, t_struct = st.tabs(
        ["Verdict", "Evidence Types", "Modality", "Stance", "Claim Structure"]
    )

    with t_dist:
        verdict_d = dists.get("verdict_label", {})
        if verdict_d:
            c_v1, c_v2 = st.columns([2, 3])
            with c_v1:
                st.bar_chart(pd.DataFrame.from_dict(verdict_d, orient="index", columns=["count"]))
            with c_v2:
                total = sum(verdict_d.values())
                for lbl, cnt in sorted(verdict_d.items(), key=lambda x: -x[1]):
                    pct = cnt / total * 100 if total else 0
                    st.markdown(f"**{lbl}** — {cnt:,}  ({pct:.1f}%)")

    with t_types:
        ev_types = dists.get("evidence_types_all", {})
        if ev_types:
            st.bar_chart(pd.DataFrame.from_dict(ev_types, orient="index", columns=["count"]))
        # vs ADR-006 targets
        tv = training.get("evidence_type_vs_targets", {})
        if tv:
            st.markdown("**vs ADR-006 Targets**")
            rows = [
                {"Type": t, "Actual": v["actual"], "Target": v["target"],
                 "Delta": f"+{v['delta']}" if v["delta"] >= 0 else str(v["delta"]),
                 "%": f"{v['pct']:.1f}"}
                for t, v in tv.items()
            ]
            st.dataframe(pd.DataFrame(rows).set_index("Type"), width='stretch')

    with t_mod:
        mod_d = dists.get("evidence_modality", {})
        if mod_d:
            st.bar_chart(pd.DataFrame.from_dict(mod_d, orient="index", columns=["count"]))

    with t_stance:
        stance_d = dists.get("evidence_stance", {})
        if stance_d:
            st.bar_chart(pd.DataFrame.from_dict(stance_d, orient="index", columns=["count"]))

    with t_struct:
        struct_d = dists.get("reasoning_structural", {})
        if struct_d:
            c_s1, c_s2 = st.columns([2, 3])
            with c_s1:
                st.bar_chart(pd.DataFrame.from_dict(struct_d, orient="index", columns=["count"]))
            with c_s2:
                total = sum(struct_d.values())
                for lbl, cnt in sorted(struct_d.items(), key=lambda x: -x[1]):
                    pct = cnt / total * 100 if total else 0
                    st.markdown(f"**{lbl}** — {cnt:,}  ({pct:.1f}%)")

    # ── Plots from reports dir ────────────────────────────────────────────────
    plots_dir = DATA_REPORT_DIR / "plots"
    if plots_dir.exists():
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            st.markdown("#### Report Plots")
            cols = st.columns(min(len(plot_files), 3))
            for i, p in enumerate(plot_files):
                cols[i % 3].image(str(p), caption=p.stem.replace("_", " "), width='stretch')

    # ── GNN readiness ─────────────────────────────────────────────────────────
    struct_d2 = dists.get("reasoning_structural", {})
    absence_n = struct_d2.get("absence", 0)
    total_r   = counts.get("total_records", 0)
    if absence_n and total_r:
        st.markdown("#### GNN Readiness")
        st.markdown(
            f"Absence claims (non_apprehension): **{absence_n:,}** ({absence_n/total_r*100:.1f}%)  \n"
            f"Zero-evidence records: **{cov.get('zero_evidence_records', 0)}**  \n"
            f"Refuted-by-absence records: **{cov.get('refuted_absence_records', 0)}**"
        )

    # ── Schema errors / logic warnings ───────────────────────────────────────
    if schema_errs or logic_warns:
        st.markdown("#### Issues")
        c_se, c_lw = st.columns(2)
        with c_se:
            st.markdown("**Schema Errors (top)**")
            for msg, cnt in list(schema_errs.items())[:8]:
                st.caption(f"`{cnt}×`  {msg}")
        with c_lw:
            st.markdown("**Logic Warnings (top)**")
            for msg, cnt in list(logic_warns.items())[:8]:
                st.caption(f"`{cnt}×`  {msg}")

    # ── Source split (training) ───────────────────────────────────────────────
    src_split = training.get("source_distribution", {})
    if src_split:
        st.markdown("#### Training Source Split")
        total_t = sum(src_split.values())
        for src, cnt in sorted(src_split.items(), key=lambda x: -x[1]):
            pct = cnt / total_t * 100 if total_t else 0
            st.markdown(f"**{src}** — {cnt:,}  ({pct:.1f}%)")


# ── Schema viewer section ─────────────────────────────────────────────────────

def _render_schema() -> None:
    try:
        from src.epistemic.schema import CLAIM_SCHEMA
        schema = CLAIM_SCHEMA
    except Exception as e:
        st.error(f"Could not load schema: {e}")
        return

    st.markdown(f"**Epistemic FactKG Schema**  v{schema.get('properties', {}).get('schema_version', {}).get('const', '?')}")
    st.caption("Unified v3.0 record schema — single source of truth in src/epistemic/schema.py")

    required = schema.get("required", [])
    props    = schema.get("properties", {})

    rows = []
    for field, defn in props.items():
        ftype  = defn.get("type", "—")
        if isinstance(ftype, list):
            ftype = " | ".join(ftype)
        fdesc  = defn.get("description", "")[:120]
        freq   = "✓" if field in required else ""
        rows.append({"Field": field, "Type": ftype, "Required": freq, "Description": fdesc})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("Field"), width='stretch', height=420)

    with st.expander("Raw JSON Schema"):
        st.json(schema)


# ── Claim search section ──────────────────────────────────────────────────────

def _render_claim_search() -> None:
    all_records = load_all_records_list()
    if not all_records:
        st.info(f"Training JSONL not found.")
        return

    st.caption(f"{len(all_records):,} total records available for search")

    # Filter widgets
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            text_q = st.text_input("Claim text contains", key="ds_text_q",
                                   placeholder="e.g. Olympic Games")
        with c2:
            claim_id_q = st.text_input("Claim ID (exact or prefix)", key="ds_id_q",
                                       placeholder="e.g. averitec-train-0001")
        with c3:
            verdicts = [""] + sorted({r.get("verdict", {}).get("label", "") for r in all_records if r.get("verdict")})
            verdict_q = st.selectbox("Verdict", verdicts, key="ds_verdict_q")

        c4, c5, c6 = st.columns(3)
        with c4:
            sources = [""] + sorted({r.get("provenance", {}).get("dataset", "") for r in all_records if r.get("provenance")})
            source_q = st.selectbox("Source dataset", sources, key="ds_source_q")
        with c5:
            structures = [""] + sorted({
                (r.get("reasoning") or {}).get("structural") or ""
                for r in all_records
            } - {""})
            struct_q = st.selectbox("Reasoning structure", structures, key="ds_struct_q")
        with c6:
            max_results = st.number_input("Max results", 5, 200, 25, 5, key="ds_max_r")

    run_search = st.button("Search", type="primary", key="ds_run")
    if not run_search:
        return

    # Apply filters
    filtered = all_records
    if text_q.strip():
        q = text_q.strip().lower()
        filtered = [r for r in filtered if q in r.get("claim", "").lower()]
    if claim_id_q.strip():
        q = claim_id_q.strip().lower()
        filtered = [r for r in filtered if r.get("id", "").lower().startswith(q)]
    if verdict_q:
        filtered = [r for r in filtered if r.get("verdict", {}).get("label") == verdict_q]
    if source_q:
        filtered = [r for r in filtered if r.get("provenance", {}).get("dataset") == source_q]
    if struct_q:
        filtered = [r for r in filtered
                    if (r.get("reasoning") or {}).get("structural") == struct_q]

    st.caption(f"Found {len(filtered):,} matching records — showing first {max_results}")
    shown = filtered[:max_results]

    for rec in shown:
        verdict_label = rec.get("verdict", {}).get("label", "?")
        claim_id      = rec.get("id", "—")
        dataset       = rec.get("provenance", {}).get("dataset", "?")
        n_ev          = len(rec.get("evidence", []))
        icons = {"supported": "✓", "refuted": "✗", "not_enough_evidence": "~", "conflicting_evidence": "⚡"}
        icon = icons.get(verdict_label, "?")

        with st.expander(
            f"{icon} `{claim_id}` [{dataset}]  ·  {rec.get('claim', '')[:80]}…",
            expanded=False,
        ):
            c_l, c_r = st.columns([3, 1])
            with c_l:
                st.markdown(f"**Claim:** {rec.get('claim', '')}")
                st.markdown(f"**Verdict:** `{verdict_label}`")
                just = rec.get("verdict", {}).get("justification", "")
                if just:
                    st.caption(f"Justification: {just[:200]}")
                # Epistemic properties
                ep = rec.get("epistemic") or {}
                reasoning = rec.get("reasoning") or {}
                ep_parts = []
                if reasoning.get("structural"):
                    ep_parts.append(f"structure: `{reasoning['structural']}`")
                if reasoning.get("strategy"):
                    ep_parts.append(f"strategy: `{reasoning['strategy']}`")
                if ep.get("assignment_method"):
                    ep_parts.append(f"method: `{ep['assignment_method']}`")
                if ep_parts:
                    st.caption("Epistemic: " + "  ·  ".join(ep_parts))
                # Provenance
                prov = rec.get("provenance") or {}
                if prov:
                    prov_parts = [f"dataset: `{prov.get('dataset','?')}`"]
                    if prov.get("original_id"):
                        prov_parts.append(f"orig_id: `{prov['original_id']}`")
                    st.caption("Provenance: " + "  ·  ".join(prov_parts))
            with c_r:
                if st.button("Load in Verify", key=f"load_{claim_id}", width='stretch'):
                    load_record_into_state(rec)
                    st.success("Loaded — switch to Verify tab")
                st.caption(f"{n_ev} evidence items")

            # Evidence preview
            evs = rec.get("evidence", [])
            if evs:
                st.markdown(f"**Evidence** ({len(evs)} items — showing first 3):")
                for j, ev in enumerate(evs[:3]):
                    st_score = ev.get("source_trust")
                    is_score = ev.get("inference_strength")
                    meta = ""
                    if st_score is not None:
                        meta += f" ST={st_score:.2f}"
                    if is_score is not None:
                        meta += f" IS={is_score:.2f}"
                    st.markdown(
                        f"**ev{j+1}** `{', '.join(ev.get('evidence_types', []) or ['?'])}` "
                        f"· `{ev.get('stance', '?')}`{meta}  \n"
                        f"{(ev.get('text') or '')[:160]}"
                    )

            # Full JSON
            with st.expander("Full JSON record", expanded=False):
                st.json(rec)


# ── Graph Browser ─────────────────────────────────────────────────────────────



def _render_graph_browser() -> None:
    """Browse HeteroData graphs from the pkl split cache."""
    import base64
    from _constants import MODELS
    from _ui import build_pyvis_html

    st.markdown(
        "Browse the actual `HeteroData` graphs stored in the split-cache pickle.  "
        "These are the graphs fed into the GNN during training / evaluation."
    )

    if not GRAPH_CACHE_DIR.exists():
        st.info(
            f"Graph cache directory `{GRAPH_CACHE_DIR}` not found.  "
            "Run training to generate the cache."
        )
        return

    available_models = [
        m for m in MODELS
        if (GRAPH_CACHE_DIR / f"split_cache_{m}.pkl").exists()
    ]
    if not available_models:
        st.info("No `split_cache_*.pkl` files found in `out/model/graphs/`.")
        return

    col_m, col_s, col_i = st.columns([2, 1, 3])
    with col_m:
        model_key = st.selectbox("Model", available_models, key="gb_model")
    with col_s:
        split = st.radio("Split", ["train", "val"], horizontal=True, key="gb_split")

    cache = load_graph_cache(model_key)
    if cache is None:
        st.warning("Failed to load cache.")
        return

    graphs = cache.get(split, [])
    if not graphs:
        st.warning(f"No graphs for split '{split}'.")
        return

    with col_i:
        idx = st.slider(
            "Graph index", 0, len(graphs) - 1, 0, key="gb_idx",
            help=f"{len(graphs)} graphs in '{split}' split",
        )

    g = graphs[idx]

    # Verdict label from claim.y
    try:
        claim_y = int(g["claim"].y[0].item())
        verdict_lbl = INT_TO_VERDICT.get(claim_y, str(claim_y))
    except Exception:
        verdict_lbl = "?"

    n_claim = g["claim"].x.shape[0]
    n_ev    = g["evidence"].x.shape[0]

    # Try to find a matching claim record
    try:
        id_map  = build_graph_id_map(model_key)
        rev_map = {v: k for k, v in id_map.items()}
        matched_cid = rev_map.get((split, idx))
    except Exception:
        matched_cid = None

    claim_text = ""
    if matched_cid:
        records = load_all_records_list()
        for rec in records:
            if rec.get("id") == matched_cid:
                claim_text = rec.get("claim", "")
                break

    st.markdown(
        f"**Split:** `{split}` \u00b7 "
        f"**Index:** {idx}/{len(graphs)-1} \u00b7 "
        f"**Verdict (GT):** `{verdict_lbl}` \u00b7 "
        f"**Nodes:** {n_claim} claim + {n_ev} evidence"
        + (f"  \u00b7  **Claim ID:** `{matched_cid}`" if matched_cid else "")
    )
    if claim_text:
        st.markdown(f"> {claim_text}")

    html = build_pyvis_html(g, claim_text=claim_text, height="520px")
    html_b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{html_b64}", height=540)

    with st.expander("Raw HeteroData repr", expanded=False):
        st.code(str(g), language=None)


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    stats = load_dataset_stats()

    t_stats, t_schema, t_search, t_graphs = st.tabs(
        ["Statistics & Insights", "Schema", "Claim Search", "Graph Browser"]
    )

    with t_stats:
        if stats is None:
            st.info("No validation data found. Run `just validate` to generate it.")
        else:
            _render_stats(stats)

    with t_schema:
        _render_schema()

    with t_search:
        _render_claim_search()

    with t_graphs:
        _render_graph_browser()
