"""Source Trust Registry tab — searchable, sortable registry viewer."""
from __future__ import annotations

import streamlit as st

from _loaders import load_registry


def render() -> None:
    records = load_registry()
    if not records:
        st.info("Registry file not found at `data/registry/source_trust_registry.jsonl`")
        return

    import pandas as pd
    df_all = pd.DataFrame(records)
    st.caption(f"{len(records)} sources registered") 

    # ── Summary bar chart (static — not affected by filters) ──────────────────
    c_chart, c_stats = st.columns([3, 1])
    with c_chart:
        st.markdown("#### Source Trust Distribution  _(all sources)_")
        st.bar_chart(df_all["source_trust"].round(2).value_counts().sort_index())
    with c_stats:
        st.markdown("&nbsp;")
        st.metric("Mean ST",   f"{df_all['source_trust'].mean():.3f}")
        st.metric("Median ST", f"{df_all['source_trust'].median():.3f}")
        st.metric("Sources",   str(len(records)))

    st.markdown("---")

    # ── Filters (above the table) ──────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            search_q = st.text_input("Search source_id / domain / name", key="reg_search",
                                     placeholder="e.g. reuters, academic, .gov")
        with c2:
            all_types = sorted({r.get("source_type", "?") for r in records})
            sel_type  = st.multiselect("Source Type", all_types, key="reg_type",
                                       placeholder="All types")
        with c3:
            all_mods = sorted({r.get("modality", "?") for r in records})
            sel_mod  = st.multiselect("Modality", all_mods, key="reg_modality",
                                      placeholder="All modalities")
        st_min, st_max = st.slider(
            "Source Trust range", 0.0, 1.0, (0.0, 1.0), 0.05, key="reg_trust_range"
        )

    # ── Filter logic ───────────────────────────────────────────────────────────
    filtered = records
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [
            r for r in filtered
            if q in r.get("source_id", "").lower()
            or q in r.get("domain", "").lower()
            or q in r.get("source_name", "").lower()
        ]
    if sel_type:
        filtered = [r for r in filtered if r.get("source_type") in sel_type]
    if sel_mod:
        filtered = [r for r in filtered if r.get("modality") in sel_mod]
    filtered = [
        r for r in filtered
        if st_min <= r.get("source_trust", 0.0) <= st_max
    ]

    st.caption(f"Showing **{len(filtered)}** / {len(records)} entries after filters")

    # ── Table ──────────────────────────────────────────────────────────────────
    if not filtered:
        st.warning("No entries match the current filters.")
        return

    rows = []
    for r in sorted(filtered, key=lambda x: x.get("source_trust", 0), reverse=True):
        rows.append({
            "source_id":        r.get("source_id", ""),
            "name":             r.get("source_name", ""),
            "domain":           r.get("domain", ""),
            "type":             r.get("source_type", ""),
            "modality":         r.get("modality", ""),
            "trust":            r.get("source_trust", 0.0),
            "prior_trust":      r.get("prior_trust", 0.0),
            "default_IS":       r.get("default_inference_strength", 0.0),
            "evidence_types":   ", ".join(r.get("default_evidence_types", [])),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["trust", "prior_trust", "default_IS"],
                                     cmap="RdYlGn", vmin=0, vmax=1),
        width='stretch',
        height=min(600, 38 * len(rows) + 40),
    )

    # ── Detail expanders ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Full Entry Detail")
    for r in sorted(filtered, key=lambda x: x.get("source_trust", 0), reverse=True)[:20]:
        trust = r.get("source_trust", 0.0)
        color = "🟢" if trust >= 0.8 else "🟡" if trust >= 0.6 else "🔴"
        with st.expander(
            f"{color} `{r['source_id']}` — {r.get('source_name', '')}  ·  ST={trust:.2f}",
            expanded=False,
        ):
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown(f"**Domain:** `{r.get('domain', '—')}`")
                st.markdown(f"**Type:** `{r.get('source_type', '—')}`")
                st.markdown(f"**Modality:** `{r.get('modality', '—')}`")
                st.markdown(f"**Evidence types:** `{', '.join(r.get('default_evidence_types', []))}`")
            with c_right:
                st.metric("Source Trust",  f"{r.get('source_trust', 0):.3f}")
                st.metric("Prior Trust",   f"{r.get('prior_trust',  0):.3f}")
                st.metric("Default IS",    f"{r.get('default_inference_strength', 0):.3f}")
            meta = r.get("trust_metadata", {})
            if meta:
                st.caption(f"Methodology: `{meta.get('methodology_ref', '—')}`  ·  "
                           f"Version: `{meta.get('trust_version', '—')}`")
