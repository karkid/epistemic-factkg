"""Verify tab — claim input, inline ID lookup, evidence cards, results + graph."""
from __future__ import annotations

import streamlit as st

from _constants import MODELS, ALL_KEY
from _loaders import get_predictor
from _state import model_selector, load_random_example, load_by_id
from _ui import (
    render_verdict_card, render_decision_path, render_arch_flow,
    render_layerwise, render_debug_view, render_evidence_cards,
    render_compare_results, build_claim_dot, build_model_computation_dot,
    build_pyvis_html,
)


def render() -> None:
    # ── Model selector ─────────────────────────────────────────────────────────
    verify_model = model_selector("verify_model_sel", allow_all=True)
    st.markdown("---")

    # ── Claim ID row + claim textarea ──────────────────────────────────────────
    current_id = st.session_state.get("current_claim_id")
    id_display = f'<span class="claim-id-badge">{current_id}</span>' if current_id else ""

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
        f'<span style="font-weight:600;font-size:0.9rem">Claim</span>'
        f'{id_display}'
        f'</div>',
        unsafe_allow_html=True,
    )

    c_claim, c_btn = st.columns([5, 1])
    with c_claim:
        claim = st.text_area(
            "Claim",
            value=st.session_state.get("last_claim", ""),
            height=80,
            placeholder="e.g.  The Eiffel Tower is in Berlin.",
            key="claim_input",
            label_visibility="collapsed",
        )
    with c_btn:
        verify = st.button("Verify", type="primary", width='stretch',
                           disabled=not claim.strip())
        if st.button("Random", width='stretch'):
            load_random_example()
            st.rerun()
        if st.session_state.get("_random_true_label"):
            st.caption(f"True: **{st.session_state._random_true_label}**")

    # ── Inline ID lookup ───────────────────────────────────────────────────────
    c_id, c_go = st.columns([4, 1])
    with c_id:
        search_id = st.text_input(
            "Load by claim ID",
            key="claim_id_input",
            placeholder="Claim ID or row index  (e.g. averitec-train-000042)",
            label_visibility="visible",
        )
    with c_go:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Load →", key="load_by_id_btn", width='stretch') and search_id.strip():
            if load_by_id(search_id.strip()):
                st.rerun()
            else:
                st.error(f"ID `{search_id.strip()}` not found.")

    st.markdown("---")
    render_evidence_cards()

    if not (verify and claim.strip()):
        return

    # ── Run model(s) ───────────────────────────────────────────────────────────
    filled = [ev for ev in st.session_state.evidence_list if ev["text"].strip()] \
             or st.session_state.evidence_list

    if verify_model == ALL_KEY:
        results: dict[str, dict | str] = {}
        with st.spinner("Running all models…"):
            for mk in list(MODELS.keys()):
                pred = get_predictor(mk)
                if isinstance(pred, str):
                    results[mk] = pred
                else:
                    try:
                        results[mk] = pred.predict(claim.strip(), filled)
                    except Exception as exc:
                        results[mk] = str(exc)
        st.markdown("---")
        st.markdown("**All Models**")
        render_compare_results(results)
        return

    pred = get_predictor(verify_model)
    if isinstance(pred, str):
        st.error(pred)
        return

    with st.spinner("Running…"):
        try:
            result = pred.predict(claim.strip(), filled)
        except Exception as exc:
            st.error(str(exc))
            return

    st.markdown("---")
    render_verdict_card(result)
    render_decision_path(result)

    t_arch, t_graph, t_model_flow, t_table, t_debug = st.tabs(
        ["Architecture Flow", "Claim Graph", "Model Flow", "Layer Table", "Debug"]
    )
    with t_arch:
        render_arch_flow(result, verify_model)
    with t_graph:
        _render_graph(claim.strip(), result)
    with t_model_flow:
        _render_gnn_flow(claim.strip(), result, verify_model)
    with t_table:
        render_layerwise(result, verify_model)
    with t_debug:
        render_debug_view(result, claim.strip())


def _render_gnn_flow(claim: str, result: dict, model_name: str) -> None:
    """Layered computation-graph showing Model inference pipeline."""
    st.caption(
        "**Computation DAG** — how the Model processes each evidence item from raw input to verdict.  \n"
        "**L0** Raw inputs (text, ST) → **L1** Encoder → **L2** H1 StanceHead → "
        "**L3** H2 IS Head → **L4** EC Formula → **L5** Aggregation → **L6** Decision → **L7** Verdict"
    )
    dot_src = build_model_computation_dot(result, model_name)
    try:
        st.graphviz_chart(dot_src, width='stretch')
    except Exception:
        st.code(dot_src, language=None)

    with st.expander("DOT source"):
        st.code(dot_src, language=None)


def _render_graph(claim: str, result: dict) -> None:
    """Interactive pyvis graph of the actual HeteroData used for inference."""
    import base64

    hetero_data = result.get("hetero_data")
    ev_texts = [ev.get("text", "") for ev in result.get("evidence_breakdown", [])]

    if hetero_data is not None:
        st.caption(
            "Actual `HeteroData` graph passed to the GNN.  "
            "Hover over nodes/edges for feature details.  "
            "**Blue** = CLAIM · **green** = supports · **red** = refutes · **gray** = neutral  \n"
            r"**Solid blue** edges = has\_evidence (CLAIM→EV)  ·  "
            r"**Orange dashed** = connected\_to (EV→CLAIM)  ·  "
            r"**Gray dashed** = co\_evidence (EV↔EV)"
        )
        html = build_pyvis_html(hetero_data, claim, ev_texts)
        html_b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
        st.iframe(f"data:text/html;base64,{html_b64}", height=560)
    else:
        # Fallback: static DOT graph
        st.caption("pyvis unavailable — showing static DOT graph.")
        dot_src = build_claim_dot(claim, result)
        try:
            st.graphviz_chart(dot_src, width='stretch')
        except Exception:
            st.code(dot_src, language=None)

    with st.expander("DOT fallback view", expanded=False):
        dot_src = build_claim_dot(claim, result)
        try:
            st.graphviz_chart(dot_src, width='stretch')
        except Exception:
            st.code(dot_src, language=None)
