"""Streamlit demo — Epistemic FactKG claim verifier."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import streamlit as st

from predictor import EpistemicPredictor

# ── Constants ─────────────────────────────────────────────────────────────────

_MODELS = {
    "v3-nli":   "v3-nli  —  NLI + Hybrid (Best)",
    "v2-hgnn":  "v2-hgnn  —  Hybrid",
    "v1-hgnn":  "v1-hgnn  —  Pure Symbolic",
    "baseline": "baseline  —  No EC Formula",
}
_ALL_KEY = "all"

_VERDICT_META = {
    "supported":           ("🟢", "SUPPORTED",           "#1a7a4a"),
    "refuted":             ("🔴", "REFUTED",              "#c0392b"),
    "not_enough_evidence": ("🟡", "NOT ENOUGH EVIDENCE",  "#7d6608"),
}
_VERDICT_LABELS   = ["supported", "refuted", "not_enough_evidence"]
_STANCE_ICON      = {"supports": "🟢", "refutes": "🔴", "neutral": "⚪"}

_MODALITIES = ["web_text", "pdf", "image", "video", "audio", "web_table"]
_MODALITY_LABELS = {
    "web_text":  "Web Text",
    "pdf":       "PDF",
    "image":     "Image",
    "video":     "Video",
    "audio":     "Audio",
    "web_table": "Table",
}
_PRAMANA_SHORT = {
    "web_text":  "Shabda",
    "pdf":       "Shabda",
    "image":     "Pratyaksha",
    "video":     "Pratyaksha",
    "audio":     "Pratyaksha",
    "web_table": "Upamana",
}
_SOURCE_TYPES  = ["unknown", "news", "academic", "government", "social_media"]
_SOURCE_LABELS = {
    "unknown":      "Unknown",
    "news":         "News",
    "academic":     "Academic",
    "government":   "Government",
    "social_media": "Social Media",
}

_DATA_JSONL = Path("out/data/training/epistemic_factkg_training.jsonl")
_TEST_IDX   = Path("out/data/splits/test_indices.json")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_test_records() -> list[dict]:
    if not _DATA_JSONL.exists() or not _TEST_IDX.exists():
        return []
    with open(_TEST_IDX) as f:
        indices = set(json.load(f)["indices"])
    records = []
    with open(_DATA_JSONL) as f:
        for i, line in enumerate(f):
            if i in indices:
                records.append(json.loads(line))
    return records


@st.cache_resource(show_spinner="Loading model…")
def _get_predictor(model_name: str) -> EpistemicPredictor | str:
    try:
        return EpistemicPredictor(model_name)
    except FileNotFoundError as e:
        return str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Epistemic FactKG")
        st.caption("Neuro-symbolic fact verification")
        st.divider()

        st.markdown("**Model**")
        model_keys    = list(_MODELS.keys()) + [_ALL_KEY]
        model_display = list(_MODELS.values()) + ["All Models (compare)"]
        idx = st.radio(
            "model_radio",
            range(len(model_keys)),
            format_func=lambda i: model_display[i],
            label_visibility="collapsed",
        )

        st.divider()
        with st.expander("EC Formula"):
            st.code("EC = 1 − (1 − ST)^(EW × IS)", language=None)
            st.caption("ST = Source Trust · EW = Evidence Weight · IS = Inference Strength")

        with st.expander("Pramana"):
            st.markdown(
                "| Modality | Sanskrit |\n|---|---|\n"
                "| Web / PDF | Shabda |\n"
                "| Image / Video | Pratyaksha |\n"
                "| Table | Upamana |"
            )

        with st.expander("Model accuracy"):
            st.markdown(
                "| Model | Acc | F1 |\n|---|---|---|\n"
                "| v3-nli | **0.815** | **0.820** |\n"
                "| v2-hgnn | 0.799 | 0.807 |\n"
                "| baseline | 0.795 | 0.802 |\n"
                "| v1-hgnn | 0.712 | 0.703 |"
            )

    return model_keys[idx]


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "evidence_list":      [_blank_ev()],
        "last_claim":         "",
        "_random_true_label": None,
        "eval_rows":          None,   # list[dict] from last eval run
        "eval_model":         None,
        "inspect_idx":        None,   # index into eval_rows for inspection
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _blank_ev() -> dict:
    return {"text": "", "source_type": "unknown", "modality": "web_text"}


def _load_random_example() -> None:
    records = _load_test_records()
    if not records:
        st.warning("Test data not found.")
        return
    rec = random.choice(records)

    # Clear stale widget keys so Streamlit picks up new values
    old_n = len(st.session_state.get("evidence_list", []))
    for i in range(old_n + 6):
        for prefix in ("ev_", "mod_", "src_"):
            st.session_state.pop(f"{prefix}{i}", None)

    new_evs = [
        {
            "text":        ev.get("text", ""),
            "source_type": "academic" if "wikipedia" in ev.get("source_id", "") else "unknown",
            "modality":    ev.get("modality", "web_text"),
        }
        for ev in rec.get("evidence", [])[:4]
    ] or [_blank_ev()]

    # Set widget key directly — value= is only used on first render
    st.session_state["claim_input"]      = rec["claim"]
    st.session_state["last_claim"]       = rec["claim"]
    st.session_state["evidence_list"]    = new_evs
    st.session_state["_random_true_label"] = rec["verdict"]["label"]
    for i, ev in enumerate(new_evs):
        st.session_state[f"ev_{i}"]  = ev["text"]
        st.session_state[f"mod_{i}"] = ev["modality"]
        st.session_state[f"src_{i}"] = ev["source_type"]


# ── Evidence card rendering ───────────────────────────────────────────────────

def _render_evidence_cards() -> None:
    c_add, c_hint = st.columns([1, 5])
    with c_add:
        if st.button("＋ Add", use_container_width=True):
            st.session_state.evidence_list.append(_blank_ev())
            st.rerun()
    with c_hint:
        st.caption(f"{len(st.session_state.evidence_list)} evidence item(s)")

    for i, ev in enumerate(st.session_state.evidence_list):
        with st.container(border=True):
            c_hd, c_rm = st.columns([11, 1])
            with c_hd:
                st.caption(f"Evidence {i + 1}")
            with c_rm:
                if st.button("✕", key=f"rm_{i}"):
                    st.session_state.evidence_list.pop(i)
                    st.rerun()

            st.session_state.evidence_list[i]["text"] = st.text_area(
                "text", value=ev["text"], height=80, key=f"ev_{i}",
                placeholder="Paste evidence text…", label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            with c1:
                mod = st.selectbox(
                    "Modality", _MODALITIES, key=f"mod_{i}",
                    index=_MODALITIES.index(ev.get("modality", "web_text")),
                    format_func=lambda m: f"{_PRAMANA_SHORT[m]} · {_MODALITY_LABELS[m]}",
                )
                st.session_state.evidence_list[i]["modality"] = mod
            with c2:
                src = st.selectbox(
                    "Source", _SOURCE_TYPES, key=f"src_{i}",
                    index=_SOURCE_TYPES.index(ev.get("source_type", "unknown")),
                    format_func=lambda s: _SOURCE_LABELS[s],
                )
                st.session_state.evidence_list[i]["source_type"] = src


# ── Architecture flow ─────────────────────────────────────────────────────────

def _node(title: str, body_md: str) -> None:
    """Render one layer node: bordered box with title + computed values."""
    with st.container(border=True):
        st.caption(title)
        st.markdown(body_md)


def _arrow(label: str = "") -> None:
    txt = f"&nbsp;&nbsp;&nbsp;&nbsp;↓ &nbsp; *{label}*" if label else "&nbsp;&nbsp;&nbsp;&nbsp;↓"
    st.markdown(txt, unsafe_allow_html=True)


def _render_arch_flow(result: dict, model_key: str) -> None:
    """Visual NN architecture flow with actual computed values at each layer."""
    is_nli    = model_key == "v3-nli"
    has_ec    = result["has_ec"]
    breakdown = result["evidence_breakdown"]
    n_ev      = len(breakdown)
    ev_dim    = "403d  (400 + NLI 3d)" if is_nli else "400d"

    # ── ① INPUT ──────────────────────────────────────────────────────────────
    lines = [f"**Claim** → `390d`", f"**Evidence ×{n_ev}** → `{ev_dim}`"]
    if is_nli:
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs")
            if nli:
                lines.append(
                    f"&nbsp;&nbsp;&nbsp;ev{i+1} NLI: "
                    f"entail `{nli['entailment']:.3f}` · "
                    f"contra `{nli['contradiction']:.3f}` · "
                    f"neutral `{nli['neutral']:.3f}`"
                )
    _node("① INPUT FEATURES", "\n\n".join(lines))

    _arrow("HeteroConv · GAT · 4 heads · 2 layers")

    # ── ② ENCODER OUTPUT ─────────────────────────────────────────────────────
    _node(
        "② GNN ENCODER OUTPUT",
        "`claim_emb` → `256d`  ·  `ev_emb ×" + str(n_ev) + "` → `256d`\n\n"
        "*Claim info flows into ev_emb via has_evidence / connected_to edges.*",
    )

    _arrow("parallel heads")

    # ── ③ H1 ‖ H2 ─────────────────────────────────────────────────────────
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        rows = ["| ev | stance | conf |", "|---|---|---|"]
        for i, ev in enumerate(breakdown):
            icon = _STANCE_ICON.get(ev["stance"], "⚪")
            rows.append(f"| ev{i+1} | {icon} {ev['stance']} | `{ev['stance_confidence']:.0%}` |")
        _node("③a STANCE HEAD (H1)  Linear(256→3)", "\n".join(rows))
    with c_h2:
        rows = ["| ev | IS pred |", "|---|---|"]
        for i, ev in enumerate(breakdown):
            rows.append(f"| ev{i+1} | `{ev['is_score']:.3f}` |")
        _node("③b IS HEAD (H2)  Linear(256→1)", "\n".join(rows))

    # ── ④ NLI bypass (v3-nli only) ───────────────────────────────────────────
    if is_nli:
        _arrow("v3-nli: NLI bypasses H1 in EC formula")
        nli_rows = [
            "| ev | entail→sup | contra→ref | neutral | EC stance |",
            "|---|---|---|---|---|",
        ]
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs") or {}
            sup_p = nli.get("entailment", 0)
            ref_p = nli.get("contradiction", 0)
            neu_p = nli.get("neutral", 0)
            best  = max(("sup", sup_p), ("ref", ref_p), ("neu", neu_p), key=lambda x: x[1])
            icon  = {"sup": "🟢", "ref": "🔴", "neu": "⚪"}.get(best[0], "⚪")
            nli_rows.append(
                f"| ev{i+1} | `{sup_p:.3f}` | `{ref_p:.3f}` | `{neu_p:.3f}` | {icon} `{best[0]}` |"
            )
        _node(
            "④ NLI CROSS-ENCODER BYPASS  (frozen DeBERTa-v3-small)",
            "\n".join(nli_rows) + "\n\n"
            "*Reorder: [contra, entail, neutral] → [ref, sup, neutral] for EC formula.*",
        )

    # ── ⑤ EC FORMULA ─────────────────────────────────────────────────────────
    if has_ec:
        num = "⑤" if not is_nli else "⑤"
        _arrow()
        ec_rows = [
            "| ev | ST | EW | IS | EC = 1−(1−ST)^(EW×IS) |",
            "|---|---|---|---|---|",
        ]
        for i, ev in enumerate(breakdown):
            ec_rows.append(
                f"| ev{i+1} | `{ev['source_trust']:.2f}` | `{ev['evidence_weight']:.2f}` "
                f"| `{ev['is_score']:.3f}` | **`{ev['ec_score']:.3f}`** |"
            )
        _node(f"{num} EC FORMULA  EC = 1 − (1−ST)^(EW × IS)", "\n".join(ec_rows))

    # ── ⑥ EC AGGREGATION ─────────────────────────────────────────────────────
    if has_ec:
        _arrow("1 − ∏(1 − EC_i × p_stance_i)  across all evidence")
        sup = result["support_score"]
        ref = result["refute_score"]
        agg_num = "⑥"
        with st.container(border=True):
            st.caption(f"{agg_num} EC AGGREGATION")
            ca, cb = st.columns(2)
            with ca:
                st.progress(min(sup, 1.0), text=f"🟢 support  `{sup:.3f}`")
            with cb:
                st.progress(min(ref, 1.0), text=f"🔴 refute   `{ref:.3f}`")

    # ── ⑦ VERDICT HEAD ───────────────────────────────────────────────────────
    vh_num = "⑦" if has_ec else "④"
    _arrow()
    if has_ec:
        head_desc = (
            "cat([scores `2d`, claim_emb `256d`]) → `258d`\n\n"
            "Linear(258→128) → ReLU → Dropout → Linear(128→3)"
        )
    else:
        head_desc = "Linear(256→128) → ReLU → Dropout → Linear(128→3)"

    probs = result["verdict_probs"]
    prob_lines = []
    for lbl, p in zip(_VERDICT_LABELS, probs):
        icon = _VERDICT_META[lbl][0]
        bar  = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        prob_lines.append(f"{icon} `{lbl}` &nbsp; `{bar}` &nbsp; `{p:.1%}`")

    _node(
        f"{vh_num} VERDICT HEAD",
        head_desc + "\n\n" + "\n\n".join(prob_lines),
    )

    # ── ⑧ FINAL VERDICT ──────────────────────────────────────────────────────
    _arrow()
    verdict = result["verdict"]
    icon, label, color = _VERDICT_META.get(verdict, ("❓", verdict.upper(), "#888"))
    st.markdown(
        f"<div style='border:2px solid {color}; border-radius:6px; padding:12px 18px; "
        f"font-size:1.3rem; font-weight:700; color:{color}; text-align:center;'>"
        f"{icon} {label}</div>",
        unsafe_allow_html=True,
    )


# ── Layer-wise reasoning display ──────────────────────────────────────────────

def _render_layerwise(result: dict, model_key: str, true_label: str | None = None) -> None:
    """Minimal layer-by-layer reasoning trace."""
    is_nli   = (model_key == "v3-nli")
    has_ec   = result["has_ec"]
    verdict  = result["verdict"]
    v_icon   = _VERDICT_META.get(verdict, ("❓",))[0]
    breakdown = result["evidence_breakdown"]

    # ── optional true-label header ────────────────────────────────────────────
    if true_label is not None:
        t_icon = _VERDICT_META.get(true_label, ("❓",))[0]
        match = verdict == true_label
        st.markdown(
            f"{'✅' if match else '❌'} &nbsp; "
            f"True: **{t_icon} {true_label}** &nbsp;·&nbsp; "
            f"Predicted: **{v_icon} {verdict}**",
            unsafe_allow_html=True,
        )
        st.divider()

    # ── per-evidence layers ───────────────────────────────────────────────────
    for i, ev in enumerate(breakdown):
        text_preview = ev["text_short"][:100] + ("…" if len(ev["text_short"]) > 100 else "")
        st.markdown(f"**Evidence {i + 1}**  ·  *{text_preview}*")

        rows: list[tuple[str, str]] = []

        if is_nli and ev.get("nli_probs"):
            nli = ev["nli_probs"]
            rows.append((
                "① NLI cross-encoder",
                f"entail `{nli['entailment']:.1%}` &nbsp; "
                f"contra `{nli['contradiction']:.1%}` &nbsp; "
                f"neutral `{nli['neutral']:.1%}`",
            ))

        s_icon = _STANCE_ICON.get(ev["stance"], "⚪")
        rows.append((
            "② Stance head (H1)",
            f"{s_icon} **{ev['stance']}** &nbsp; conf `{ev['stance_confidence']:.0%}`",
        ))

        rows.append((
            "③ IS head (H2)",
            f"`{ev['is_score']:.3f}`",
        ))

        if has_ec:
            rows.append((
                "④ EC formula",
                f"`{ev['ec_score']:.3f}` &nbsp; "
                f"ST `{ev['source_trust']:.2f}` · EW `{ev['evidence_weight']:.2f}` · IS `{ev['is_score']:.3f}`",
            ))

        rows.append((
            "⑤ Pramana",
            f"{_PRAMANA_SHORT.get(ev['modality'], '—')} · "
            f"{_MODALITY_LABELS.get(ev['modality'], ev['modality'])} · "
            f"{_SOURCE_LABELS.get(ev['source_type'], ev['source_type'])}",
        ))

        # render as mini table
        tbl = "| Layer | Value |\n|---|---|\n"
        for label, value in rows:
            tbl += f"| {label} | {value} |\n"
        st.markdown(tbl)

        if i < len(breakdown) - 1:
            st.markdown("")

    # ── aggregation + verdict ─────────────────────────────────────────────────
    st.divider()

    if has_ec:
        sup = result["support_score"]
        ref = result["refute_score"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**⑥ EC aggregation** &nbsp; support `{sup:.3f}` / refute `{ref:.3f}`")
            st.progress(min(sup, 1.0), text=f"🟢 support {sup:.3f}")
            st.progress(min(ref, 1.0), text=f"🔴 refute  {ref:.3f}")
        with c2:
            probs = result["verdict_probs"]
            st.markdown("**⑦ Verdict head**")
            for lbl, p in zip(_VERDICT_LABELS, probs):
                icon = _VERDICT_META[lbl][0]
                st.progress(p, text=f"{icon} {p:.0%}")
    else:
        probs = result["verdict_probs"]
        st.markdown("**⑥ Verdict head**")
        for lbl, p in zip(_VERDICT_LABELS, probs):
            icon = _VERDICT_META[lbl][0]
            st.progress(p, text=f"{icon} {p:.0%}")

    st.markdown(f"**Final verdict: {v_icon} {verdict.upper().replace('_', ' ')}**")


# ── Verify tab results ────────────────────────────────────────────────────────

def _render_compare_results(results: dict[str, dict | str]) -> None:
    cols = st.columns(4)
    for col, key in zip(cols, ["baseline", "v1-hgnn", "v2-hgnn", "v3-nli"]):
        result = results.get(key)
        with col:
            st.markdown(f"**{key}**")
            if result is None:
                st.info("—")
                continue
            if isinstance(result, str):
                st.error(result[:60])
                continue
            icon, label, color = _VERDICT_META.get(result["verdict"], ("❓", "UNKNOWN", "#888"))
            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>{icon} {label}</span>",
                unsafe_allow_html=True,
            )
            for lbl, p in zip(_VERDICT_LABELS, result["verdict_probs"]):
                st.progress(p, text=f"{_VERDICT_META[lbl][0]} {p:.0%}")
            if result["has_ec"]:
                s, r = result["support_score"], result["refute_score"]
                st.caption(f"EC sup `{s:.3f}` ref `{r:.3f}`")

    st.divider()
    best = next(
        (k for k in ["v3-nli", "v2-hgnn"] if k in results and not isinstance(results[k], str)),
        None,
    )
    if best:
        st.markdown(f"*Layerwise trace — **{best}***")
        _render_layerwise(results[best], best)


# ── Evaluate tab ──────────────────────────────────────────────────────────────

def _render_evaluate_tab(selected_key: str) -> None:
    records = _load_test_records()
    if not records:
        st.warning(f"Test data not found.\n- `{_DATA_JSONL}`\n- `{_TEST_IDX}`")
        return

    models_to_eval = list(_MODELS.keys()) if selected_key == _ALL_KEY else [selected_key]

    c_n, c_seed = st.columns([3, 2])
    with c_n:
        n = st.slider("Samples", 5, min(len(records), 1000), 20, 5)
    with c_seed:
        fixed = st.checkbox("Fixed seed", value=False)
        seed  = st.number_input("Seed", value=42, step=1, label_visibility="collapsed") if fixed else None

    run = st.button("▶ Run", type="primary")

    if run:
        rng    = random.Random(seed)
        sample = rng.sample(records, min(n, len(records)))
        prog   = st.progress(0.0, text="Loading…")
        rows: list[dict] = []

        preds: dict[str, EpistemicPredictor | str] = {m: _get_predictor(m) for m in models_to_eval}
        total = len(models_to_eval) * len(sample)
        step  = 0

        for m in models_to_eval:
            pred = preds[m]
            for rec in sample:
                step += 1
                prog.progress(step / total, text=f"{m} · {step}/{total}")
                true = rec["verdict"]["label"]
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
                                 "pred": None, "result": None, "source": source,
                                 "error": str(exc)})

        prog.empty()
        st.session_state["eval_rows"]  = rows
        st.session_state["eval_model"] = models_to_eval[0] if len(models_to_eval) == 1 else _ALL_KEY
        st.session_state["inspect_idx"] = None

    rows = st.session_state.get("eval_rows")
    if not rows:
        return

    all_models = sorted({r["model"] for r in rows})

    # ── metrics ──────────────────────────────────────────────────────────────
    for m in all_models:
        mrows = [r for r in rows if r["model"] == m]
        valid = [r for r in mrows if r["pred"] is not None]
        if not valid:
            st.error(f"{m}: no valid predictions")
            continue

        correct = sum(r["pred"] == r["true"] for r in valid)
        acc     = correct / len(valid)
        tp: Counter = Counter(); fp: Counter = Counter(); fn: Counter = Counter()
        for r in valid:
            t, p = r["true"], r["pred"]
            if t == p:  tp[t] += 1
            else:       fp[p] += 1; fn[t] += 1
        f1s = []
        for lbl in _VERDICT_LABELS:
            pr = tp[lbl] / (tp[lbl] + fp[lbl]) if tp[lbl] + fp[lbl] else 0
            re = tp[lbl] / (tp[lbl] + fn[lbl]) if tp[lbl] + fn[lbl] else 0
            f1s.append(2 * pr * re / (pr + re) if pr + re else 0)
        mf1 = sum(f1s) / len(f1s)

        with st.container(border=True):
            st.markdown(f"**{m}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy",  f"{acc:.1%}")
            c2.metric("Macro F1",  f"{mf1:.3f}")
            c3.metric("n",         f"{len(valid)}/{len(mrows)}")

            fc = st.columns(3)
            for i, lbl in enumerate(_VERDICT_LABELS):
                fc[i].metric(
                    f"{_VERDICT_META[lbl][0]} F1",
                    f"{f1s[i]:.3f}",
                    help=lbl,
                )

    st.divider()

    # ── inspection table ─────────────────────────────────────────────────────
    st.markdown("**Predictions** — click Inspect to trace layer-wise reasoning")

    inspect_model = all_models[0] if len(all_models) == 1 else st.selectbox(
        "Show predictions for", all_models, label_visibility="collapsed"
    )
    mrows = [r for r in rows if r["model"] == inspect_model]

    for row in mrows[:50]:
        if row["pred"] is None:
            continue
        ok     = row["pred"] == row["true"]
        t_icon = _VERDICT_META.get(row["true"],  ("❓",))[0]
        p_icon = _VERDICT_META.get(row["pred"],  ("❓",))[0]
        marker = "✅" if ok else "❌"

        with st.expander(
            f"{marker}  {row['source']}  ·  "
            f"{row['claim'][:70]}{'…' if len(row['claim']) > 70 else ''}  "
            f"[{t_icon}→{p_icon}]",
            expanded=False,
        ):
            if row["result"] is not None:
                t_a, t_t = st.tabs(["Architecture Flow", "Layer Table"])
                with t_a:
                    _render_arch_flow(row["result"], inspect_model)
                with t_t:
                    _render_layerwise(row["result"], inspect_model, true_label=row["true"])
            else:
                st.error(row.get("error", "no result"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Epistemic FactKG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "<style>.stProgress > div > div > div { border-radius:3px; }</style>",
        unsafe_allow_html=True,
    )

    selected_key = _render_sidebar()
    _init_state()

    st.markdown("# Epistemic Claim Verifier")
    st.caption("Fact-checking grounded in the Pramana epistemic framework.")

    tab_verify, tab_eval = st.tabs(["Verify Claim", "Evaluate"])

    # ── Verify ────────────────────────────────────────────────────────────────
    with tab_verify:
        c_claim, c_btn = st.columns([5, 1])
        with c_claim:
            claim = st.text_area(
                "Claim",
                value=st.session_state.get("last_claim", ""),
                height=80,
                placeholder="e.g.  The Eiffel Tower is in Berlin.",
                key="claim_input",
            )
        with c_btn:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            verify = st.button("Verify", type="primary", use_container_width=True,
                               disabled=not claim.strip())
            if st.button("Random", use_container_width=True):
                _load_random_example()
                st.rerun()
            if st.session_state.get("_random_true_label"):
                st.caption(f"True: **{st.session_state._random_true_label}**")

        st.markdown("---")
        _render_evidence_cards()

        if verify and claim.strip():
            filled = [ev for ev in st.session_state.evidence_list if ev["text"].strip()] \
                     or st.session_state.evidence_list

            if selected_key == _ALL_KEY:
                results: dict[str, dict | str] = {}
                with st.spinner("Running all models…"):
                    for mk in list(_MODELS.keys()):
                        pred = _get_predictor(mk)
                        if isinstance(pred, str):
                            results[mk] = pred
                        else:
                            try:
                                results[mk] = pred.predict(claim.strip(), filled)
                            except Exception as exc:
                                results[mk] = str(exc)
                st.markdown("---")
                st.markdown("**All Models**")
                _render_compare_results(results)
            else:
                pred = _get_predictor(selected_key)
                if isinstance(pred, str):
                    st.error(pred)
                else:
                    with st.spinner("Running…"):
                        try:
                            result = pred.predict(claim.strip(), filled)
                        except Exception as exc:
                            st.error(str(exc))
                            return
                    st.markdown("---")
                    t_arch, t_table = st.tabs(["Architecture Flow", "Layer Table"])
                    with t_arch:
                        _render_arch_flow(result, selected_key)
                    with t_table:
                        _render_layerwise(result, selected_key)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    with tab_eval:
        st.markdown("### Evaluate on test set")
        _render_evaluate_tab(selected_key)


if __name__ == "__main__":
    main()
