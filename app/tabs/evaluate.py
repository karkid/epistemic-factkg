"""Evaluate and Compare tabs — batch evaluation on test set."""
from __future__ import annotations

import random
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components

from _constants import MODELS, ALL_KEY, VERDICT_LABELS, VERDICT_META
from _loaders import load_test_records, get_predictor
from _state import model_selector
from _ui import (
    render_arch_flow, render_layerwise, render_debug_view,
    build_claim_dot, build_model_computation_dot, build_pyvis_html,
)
from predictor import EpistemicPredictor


# ── Export helpers ────────────────────────────────────────────────────────────

_PLACEHOLDER_TEXTS = frozenset([
    "no sensor evidence found for this object type.",
    "no evidence found.",
])
_QA_PREFIXES = ("q:", "did ", "does ", "is ", "are ", "was ", "were ", "have ", "has ", "can ", "could ", "would ")


def _failure_pattern(row: dict) -> str:
    if row["pred"] is None:
        return "error"
    if row["pred"] == row["true"]:
        return "correct"
    result = row.get("result") or {}
    has_ec = result.get("has_ec", False)
    sup    = result.get("support_score", 0.0)
    ref    = result.get("refute_score",  0.0)
    if has_ec:
        if ref > 0.35 and row["pred"] != "refuted":
            return "ec_override_fix"
        if sup > 0.35 and row["pred"] != "supported":
            return "ec_override_fix"
        if max(sup, ref) < 0.20:
            return "all_neutral_ec"
    bd = result.get("evidence_breakdown") or []
    if any((ev.get("text") or "").strip().lower() in _PLACEHOLDER_TEXTS for ev in bd):
        return "placeholder_ev"
    if any((ev.get("text") or "").lower().lstrip().startswith(_QA_PREFIXES) for ev in bd):
        return "qa_format_ev"
    return "genuine_error"


def _build_eval_export(rows: list[dict]) -> str:
    import json
    export = []
    for row in rows:
        result = row.get("result") or {}
        probs  = result.get("verdict_probs") or [0.0, 0.0, 0.0]
        sup    = result.get("support_score", 0.0)
        ref    = result.get("refute_score",  0.0)
        bd     = result.get("evidence_breakdown") or []
        pred   = row.get("pred")
        true   = row.get("true", "")
        has_placeholder = any((ev.get("text") or "").strip().lower() in _PLACEHOLDER_TEXTS for ev in bd)
        has_qa = any((ev.get("text") or "").lower().lstrip().startswith(_QA_PREFIXES) for ev in bd)
        ec_disagrees = (ref > 0.35 and pred != "refuted") or (sup > 0.35 and pred != "supported")
        export.append({
            "model":              row.get("model", ""),
            "source_dataset":     row.get("source", ""),
            "claim":              row.get("claim", ""),
            "true_label":         true,
            "predicted_label":    pred,
            "correct":            pred == true if pred is not None else None,
            "support_score":      round(sup, 4),
            "refute_score":       round(ref, 4),
            "verdict_probs":      {"supported": round(probs[0], 4), "refuted": round(probs[1], 4), "nei": round(probs[2], 4)},
            "has_placeholder_ev": has_placeholder,
            "has_qa_ev":          has_qa,
            "ec_disagrees":       ec_disagrees,
            "failure_pattern":    _failure_pattern(row),
            "evidence":           [
                {"text": ev.get("text",""), "stance": ev.get("stance"),
                 "stance_confidence": ev.get("stance_confidence"),
                 "is_score": ev.get("is_score"), "source_trust": ev.get("source_trust"),
                 "evidence_weight": ev.get("evidence_weight"), "ec_score": ev.get("ec_score"),
                 "pramana": ev.get("pramana"), "source_type": ev.get("source_type"),
                 "nli_probs": ev.get("nli_probs")}
                for ev in bd
            ],
        })
    return json.dumps(export, indent=2, ensure_ascii=False)


# ── Core evaluate renderer ────────────────────────────────────────────────────

def render_evaluate(selected_key: str, *, state_key: str = "eval") -> None:
    records = load_test_records()
    if not records:
        st.warning("Test data not found.")
        return

    st.caption(f"{len(records)} test records available")
    models_to_eval = list(MODELS.keys()) if selected_key == ALL_KEY else [selected_key]

    c_n, c_seed = st.columns([3, 2])
    with c_n:
        n = st.slider("Samples", 5, min(len(records), 1000), 20, 5, key=f"{state_key}_samples")
    with c_seed:
        fixed = st.checkbox("Fixed seed", value=False, key=f"{state_key}_fixed_seed")
        seed  = st.number_input("Seed", value=42, step=1, label_visibility="collapsed",
                                key=f"{state_key}_seed") if fixed else None

    if st.button("▶ Run", type="primary", key=f"{state_key}_run_btn"):
        rng    = random.Random(seed)
        sample = rng.sample(records, min(n, len(records)))
        prog   = st.progress(0.0, text="Loading…")
        rows: list[dict] = []

        preds: dict[str, EpistemicPredictor | str] = {m: get_predictor(m) for m in models_to_eval}
        total = len(models_to_eval) * len(sample)
        step  = 0

        for m in models_to_eval:
            pred = preds[m]
            for rec in sample:
                step += 1
                prog.progress(step / total, text=f"{m} · {step}/{total}")
                true   = rec["verdict"]["label"]
                source = rec.get("provenance", {}).get("dataset", "?")
                if isinstance(pred, str):
                    rows.append({"model": m, "claim": rec["claim"], "true": true,
                                 "pred": None, "result": None, "source": source})
                    continue
                try:
                    out = pred.predict_from_record(rec)
                    rows.append({"model": m, "claim": rec["claim"], "true": true,
                                 "pred": out["verdict"], "result": out, "source": source})
                except Exception as exc:
                    rows.append({"model": m, "claim": rec["claim"], "true": true,
                                 "pred": None, "result": None, "source": source, "error": str(exc)})

        prog.empty()
        st.session_state[f"{state_key}_rows"]   = rows
        st.session_state[f"{state_key}_inspect"] = None

    rows = st.session_state.get(f"{state_key}_rows")
    if not rows:
        return

    all_models = sorted({r["model"] for r in rows})

    # ── Metrics ───────────────────────────────────────────────────────────────
    for m in all_models:
        mrows = [r for r in rows if r["model"] == m]
        valid = [r for r in mrows if r["pred"] is not None]
        if not valid:
            st.error(f"{m}: no valid predictions")
            continue

        correct = sum(r["pred"] == r["true"] for r in valid)
        acc     = correct / len(valid)
        tp: Counter = Counter()
        fp: Counter = Counter()
        fn: Counter = Counter()
        for r in valid:
            t, p = r["true"], r["pred"]
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1
        f1s = []
        for lbl in VERDICT_LABELS:
            pr = tp[lbl] / (tp[lbl] + fp[lbl]) if tp[lbl] + fp[lbl] else 0.0
            re = tp[lbl] / (tp[lbl] + fn[lbl]) if tp[lbl] + fn[lbl] else 0.0
            f1s.append(2 * pr * re / (pr + re) if pr + re else 0.0)
        mf1 = sum(f1s) / len(f1s)

        with st.container(border=True):
            st.markdown(f"**{m}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{acc:.1%}")
            c2.metric("Macro F1", f"{mf1:.3f}")
            c3.metric("n",        f"{len(valid)}/{len(mrows)}")
            fc = st.columns(3)
            for i, lbl in enumerate(VERDICT_LABELS):
                fc[i].metric(f"{VERDICT_META[lbl][0]} F1", f"{f1s[i]:.3f}", help=lbl)

    # ── Download ──────────────────────────────────────────────────────────────
    valid_rows = [r for r in rows if r.get("pred") is not None]
    if valid_rows:
        json_data = _build_eval_export(rows)
        c_dl, c_pat = st.columns([2, 5])
        with c_dl:
            st.download_button("⬇ Download JSON", json_data, file_name="eval_results.json",
                               mime="application/json", use_container_width=True,
                               key=f"{state_key}_dl_btn")
        with c_pat:
            patterns = Counter(_failure_pattern(r) for r in rows)
            parts = [f"`{p}` ×{n}" for p in
                     ["correct","ec_override_fix","all_neutral_ec","placeholder_ev",
                      "qa_format_ev","genuine_error","error"]
                     if (n := patterns.get(p, 0))]
            st.caption("  ·  ".join(parts))

    st.divider()

    # ── Inspection table ──────────────────────────────────────────────────────
    st.markdown("**Predictions** — expand to trace layer-wise reasoning")
    inspect_model = all_models[0] if len(all_models) == 1 else st.selectbox(
        "Show predictions for", all_models, label_visibility="collapsed",
        key=f"{state_key}_inspect_model",
    )
    mrows = [r for r in rows if r["model"] == inspect_model]

    for idx, row in enumerate(mrows[:50]):
        if row["pred"] is None:
            continue
        ok     = row["pred"] == row["true"]
        t_icon = VERDICT_META.get(row["true"],  ("?",))[0]
        p_icon = VERDICT_META.get(row["pred"],  ("?",))[0]
        claim_snip = row["claim"][:70] + ("…" if len(row["claim"]) > 70 else "")
        # Use unique widget key prefix per row to avoid Streamlit mixing
        row_key = f"{state_key}_{inspect_model}_{idx}"
        with st.expander(
            f"{'✅' if ok else '❌'}  {row['source']}  ·  {claim_snip}  [{t_icon}→{p_icon}]",
            expanded=False,
        ):
            if row["result"] is not None:
                t_arch, t_graph, t_flow, t_table, t_dbg = st.tabs(
                    ["Architecture Flow", "Claim Graph", "Model Flow", "Layer Table", "Debug"]
                )
                with t_arch:
                    render_arch_flow(row["result"], inspect_model)
                with t_graph:
                    hd       = row["result"].get("hetero_data")
                    ev_texts = [
                        ev.get("text", "")
                        for ev in (row["result"].get("evidence_breakdown") or [])
                    ]
                    if hd is not None:
                        st.caption(
                            "Actual `HeteroData` graph.  "
                            "Hover nodes/edges for feature details.  "
                            "**Blue** = CLAIM · **green** = supports · "
                            "**red** = refutes · **gray** = neutral"
                        )
                        components.html(
                            build_pyvis_html(hd, row["claim"], ev_texts),
                            height=520, scrolling=False,
                        )
                    else:
                        dot_src = build_claim_dot(row["claim"], row["result"])
                        try:
                            st.graphviz_chart(dot_src, use_container_width=True)
                        except Exception:
                            st.code(dot_src, language=None)
                with t_flow:
                    dot_src = build_model_computation_dot(row["result"], inspect_model)
                    try:
                        st.graphviz_chart(dot_src, use_container_width=True)
                    except Exception:
                        st.code(dot_src, language=None)
                    with st.expander("DOT source", expanded=False):
                        st.code(dot_src, language=None)
                with t_table:
                    render_layerwise(row["result"], inspect_model, true_label=row["true"])
                with t_dbg:
                    render_debug_view(row["result"], row["claim"])
            else:
                st.error(row.get("error", "no result"))


# ── Public tab renderers ──────────────────────────────────────────────────────

def render_evaluate_tab() -> None:
    eval_model = model_selector("eval_model_sel")
    st.markdown("---")
    render_evaluate(eval_model, state_key="eval")


def render_compare_tab() -> None:
    st.caption("Run all 4 models on the same random sample and compare results side-by-side.")
    st.markdown("---")
    render_evaluate(ALL_KEY, state_key="compare")
