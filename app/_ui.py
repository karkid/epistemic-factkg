"""All shared UI rendering helpers (pure render functions, no tab logic)."""
from __future__ import annotations

import streamlit as st

from _constants import (
    VERDICT_META, VERDICT_LABELS, VERDICT_CSS, STANCE_CHIP_CLS,
    MODALITY_LABELS, PRAMANA_SHORT, SOURCE_LABELS, MODALITIES, SOURCE_TYPES,
)
from _state import blank_ev


# ── Primitive helpers ─────────────────────────────────────────────────────────

def chip(text: str, cls: str = "chip-gray") -> str:
    return f'<span class="chip {cls}">{text}</span>'


def arch_box(css_class: str, step: str, title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="arch-box {css_class}">'
        f'<div class="arch-box-title">{step} &nbsp; {title}</div>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def arch_arrow(label: str = "") -> None:
    inner = (
        f"↓ &thinsp; <em style='color:var(--text-muted);font-size:0.75rem'>{label}</em>"
        if label else "↓"
    )
    st.markdown(
        f'<div style="text-align:center;color:var(--text-muted);padding:4px 0">{inner}</div>',
        unsafe_allow_html=True,
    )


def triple_label(uri: str) -> str:
    seg = uri.split("/")[-1]
    return seg.split("|")[0] or seg


def render_triple(triple: list) -> str:
    if not triple or len(triple) < 3:
        return ""
    s = triple_label(str(triple[0]))
    p = str(triple[1])
    o = triple_label(str(triple[2]))
    arrow = '<span style="color:var(--text-muted);font-size:0.75rem">→</span>'
    return (
        f'<div class="triple-row">'
        f'{chip(s, "chip-blue")}{arrow}'
        f'{chip(p, "chip-purple")}{arrow}'
        f'{chip(o, "chip-green")}'
        f"</div>"
    )


def ev_table(headers: list[str], rows_data: list[list[str]]) -> str:
    th = "".join(
        f'<th style="padding:2px 5px;text-align:{"left" if j == 0 else "center"}">{h}</th>'
        for j, h in enumerate(headers)
    )
    trs = "".join(
        "<tr>" + "".join(
            f'<td style="padding:2px 5px;text-align:{"left" if j == 0 else "center"}">{cell}</td>'
            for j, cell in enumerate(row)
        ) + "</tr>"
        for row in rows_data
    )
    return (
        '<table style="font-size:0.75rem;border-collapse:collapse;width:100%">'
        f"<tr>{th}</tr>{trs}</table>"
    )


# ── Decision path callout ─────────────────────────────────────────────────────

def render_decision_path(result: dict) -> None:
    if not result["has_ec"]:
        st.markdown(
            '<div class="decision-path dp-baseline">▶ Baseline — VerdictHead only (no EC formula)</div>',
            unsafe_allow_html=True,
        )
        return
    sup = result["support_score"]
    ref = result["refute_score"]
    _EC = 0.35
    if sup > _EC and ref > _EC:
        st.markdown(
            f'<div class="decision-path dp-conflict">⚡ Conflicting evidence — '
            f'sup {sup:.3f} & ref {ref:.3f} both >{_EC} → VerdictHead decides</div>',
            unsafe_allow_html=True,
        )
    elif sup > _EC:
        st.markdown(
            f'<div class="decision-path dp-sup">✓ Symbolic override — '
            f'EC support {sup:.3f} >{_EC} → SUPPORTED</div>',
            unsafe_allow_html=True,
        )
    elif ref > _EC:
        st.markdown(
            f'<div class="decision-path dp-ref">✗ Symbolic override — '
            f'EC refute {ref:.3f} >{_EC} → REFUTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="decision-path dp-weak">~ EC weak — '
            f'sup {sup:.3f}, ref {ref:.3f} both ≤{_EC} → VerdictHead decides</div>',
            unsafe_allow_html=True,
        )


# ── Pipeline bar ──────────────────────────────────────────────────────────────

def render_arch_pipeline_bar(model_key: str) -> None:
    stages: list[tuple[str, str]] = [
        ("#6b7280", "① Input"),
        ("#1d4ed8", "② GNN Encoder"),
        ("#b45309", "③ H1+H2"),
    ]
    if model_key == "v3-nli":
        stages.append(("#6d28d9", "④ NLI Bypass"))
    if model_key != "baseline":
        stages.append(("#15803d", "⑤⑥ EC Formula"))
    stages.append(("#991b1b", "⑦ VerdictHead"))

    def _pill(color: str, label: str) -> str:
        return (
            f'<span style="background:{color};color:#fff;padding:3px 11px;'
            f'border-radius:12px;font-size:0.72rem;font-weight:600;white-space:nowrap">{label}</span>'
        )

    pills = ' <span style="color:#9ca3af;font-size:0.85rem">→</span> '.join(
        _pill(c, lbl) for c, lbl in stages
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;'
        f'padding:10px 14px;background:#f8fafc;border:1px solid #e5e7eb;'
        f'border-radius:8px;margin-bottom:10px">{pills}</div>',
        unsafe_allow_html=True,
    )


# ── Verdict card ──────────────────────────────────────────────────────────────

def render_verdict_card(result: dict) -> None:
    verdict = result["verdict"]
    probs   = result["verdict_probs"]
    icon, label, _ = VERDICT_META.get(verdict, ("?", verdict.upper(), "#888"))
    css_cls = VERDICT_CSS.get(verdict, "nei")
    conf    = max(probs) * 100
    color_vars = ["green", "red", "amber"]
    segs = "".join(
        f'<div class="minibar-seg" style="background:var(--{c});width:{p * 100:.1f}%"></div>'
        for c, p in zip(color_vars, probs)
    )
    p0, p1, p2 = probs[0], probs[1], probs[2]
    st.markdown(
        f'<div class="verdict-card {css_cls}">'
        f'<div class="verdict-label">{icon} {label}</div>'
        f'<div class="verdict-conf">{conf:.1f}% confidence</div>'
        f'<div class="minibar-wrap" style="width:220px;margin:8px auto 0">{segs}</div>'
        f'<div style="font-size:0.70rem;margin-top:3px">'
        f"sup {p0:.0%} &middot; ref {p1:.0%} &middot; nei {p2:.0%}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ── Architecture flow ─────────────────────────────────────────────────────────

def render_arch_flow(result: dict, model_key: str) -> None:
    render_arch_pipeline_bar(model_key)
    is_nli    = model_key == "v3-nli"
    has_ec    = result["has_ec"]
    breakdown = result["evidence_breakdown"]
    n_ev      = len(breakdown)
    ev_dim    = "403d (400 + NLI 3d)" if is_nli else "400d"

    input_html = f"<div>Claim <b>390d</b> &nbsp;·&nbsp; Evidence &times;{n_ev} <b>{ev_dim}</b></div>"
    if is_nli:
        nli_rows = []
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs") or {}
            e, c, n = nli.get("entailment", 0), nli.get("contradiction", 0), nli.get("neutral", 0)
            nli_rows.append([
                f"ev{i + 1}",
                chip(f"{e:.3f}", "chip-green" if e > 0.5 else "chip-gray"),
                chip(f"{c:.3f}", "chip-red" if c > 0.5 else "chip-gray"),
                chip(f"{n:.3f}", "chip-gray"),
            ])
        input_html += ev_table(["ev", "entail", "contra", "neutral"], nli_rows)
    arch_box("", "①", "INPUT FEATURES", input_html)
    arch_arrow("HeteroConv · GAT 4 heads · 2 layers")

    arch_box(
        "enc", "②", "GNN ENCODER",
        f"<div>claim_emb <b>256d</b> &nbsp;&middot;&nbsp; ev_emb &times;{n_ev} <b>256d</b></div>"
        '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">'
        "Claim context flows into ev_emb via has_evidence / connected_to edges</div>",
    )
    arch_arrow("parallel heads")

    c_h1, c_h2 = st.columns(2)
    with c_h1:
        h1_rows = []
        for i, ev in enumerate(breakdown):
            stance = ev["stance"]
            conf   = ev["stance_confidence"]
            h1_rows.append([f"ev{i+1}", chip(stance, STANCE_CHIP_CLS.get(stance, "chip-gray")),
                             chip(f"{conf:.0%}", "chip-gray")])
        arch_box("stance", "③a", "STANCE HEAD (H1) &nbsp; Linear(256→3)",
                 ev_table(["ev", "stance", "conf"], h1_rows))
    with c_h2:
        h2_rows = []
        for i, ev in enumerate(breakdown):
            is_val = ev["is_score"]
            is_cls = "chip-green" if is_val > 0.5 else "chip-amber" if is_val > 0.3 else "chip-gray"
            h2_rows.append([f"ev{i+1}", chip(f"{is_val:.3f}", is_cls)])
        arch_box("stance", "③b", "IS HEAD (H2) &nbsp; Linear(256→1)", ev_table(["ev", "IS score"], h2_rows))

    if is_nli:
        arch_arrow("v3-nli: NLI bypasses H1 in EC formula")
        nli4 = []
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs") or {}
            sp, rf, ne = nli.get("entailment", 0), nli.get("contradiction", 0), nli.get("neutral", 0)
            best, _ = max([("sup", sp), ("ref", rf), ("neu", ne)], key=lambda x: x[1])
            ec_cls = {"sup": "chip-green", "ref": "chip-red", "neu": "chip-gray"}.get(best, "chip-gray")
            nli4.append([f"ev{i+1}",
                         chip(f"{sp:.3f}", "chip-green" if sp > 0.5 else "chip-gray"),
                         chip(f"{rf:.3f}", "chip-red" if rf > 0.5 else "chip-gray"),
                         chip(f"{ne:.3f}", "chip-gray"), chip(best, ec_cls)])
        arch_box("nli", "④", "NLI CROSS-ENCODER BYPASS &nbsp; (frozen DeBERTa-v3-small)",
                 ev_table(["ev", "entail→sup", "contra→ref", "neutral", "EC stance"], nli4)
                 + '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">'
                 "Reorder: [contra, entail, neutral] → [ref, sup, neutral] for EC formula</div>")

    if has_ec:
        arch_arrow()
        ec_rows = []
        for i, ev in enumerate(breakdown):
            ec_val = ev["ec_score"]
            ec_cls = "chip-green" if ec_val > 0.5 else "chip-amber" if ec_val > 0.2 else "chip-gray"
            ec_rows.append([f"ev{i+1}", chip(f"{ev['source_trust']:.2f}", "chip-gray"),
                             chip(f"{ev['evidence_weight']:.2f}", "chip-gray"),
                             chip(f"{ev['is_score']:.3f}", "chip-gray"), chip(f"{ec_val:.3f}", ec_cls)])
        arch_box("ec", "⑤", "EC FORMULA &nbsp; EC = 1−(1−ST)^(EW×IS)",
                 ev_table(["ev", "ST", "EW", "IS", "EC"], ec_rows))

        arch_arrow("1 − ∏(1 − ECᵢ × p_stanceᵢ)  across all evidence")
        sup, ref = result["support_score"], result["refute_score"]

        def _bar(color: str, width: float) -> str:
            return (f'<div style="background:#e9ecef;border-radius:3px;height:6px;margin:3px 0">'
                    f'<div style="background:var(--{color});height:100%;width:{min(width,100):.1f}%;border-radius:3px"></div></div>')

        bars = (
            f'<div style="margin:4px 0">{chip(f"support {sup:.3f}", "chip-green" if sup > 0.35 else "chip-gray")}{_bar("green", sup*100)}</div>'
            f'<div style="margin:4px 0">{chip(f"refute {ref:.3f}", "chip-red" if ref > 0.35 else "chip-gray")}{_bar("red", ref*100)}</div>'
        )
        arch_box("ec", "⑥", "EC AGGREGATION", bars)

    vh_num = "⑦" if has_ec else "④"
    arch_arrow()
    head_html = (
        "<div>cat([EC scores <b>2d</b>, claim_emb <b>256d</b>]) → <b>258d</b>"
        " → Linear(258→128) → ReLU → Dropout → Linear(128→3)</div>"
        if has_ec else
        "<div>Linear(256→128) → ReLU → Dropout → Linear(128→3)</div>"
    )
    probs = result["verdict_probs"]
    prob_colors = {"supported": "green", "refuted": "red", "not_enough_evidence": "amber"}
    prob_chip_cls = {"supported": "chip-green", "refuted": "chip-red", "not_enough_evidence": "chip-amber"}
    short_labels  = {"supported": "supported", "refuted": "refuted", "not_enough_evidence": "NEI"}
    prob_html = "".join(
        f'<div style="margin:4px 0;display:flex;align-items:center;gap:6px">'
        f'{chip(short_labels[lbl], prob_chip_cls[lbl])}'
        f'<div style="flex:1;background:#e9ecef;border-radius:3px;height:6px">'
        f'<div style="background:var(--{prob_colors[lbl]});height:100%;width:{p*100:.1f}%;border-radius:3px"></div></div>'
        f'<span style="font-size:0.75rem;min-width:38px">{p:.1%}</span></div>'
        for lbl, p in zip(VERDICT_LABELS, probs)
    )
    arch_box("verdict", vh_num, "VERDICT HEAD", head_html + prob_html)
    arch_arrow()
    render_verdict_card(result)


# ── Layer-wise reasoning display ──────────────────────────────────────────────

def render_layerwise(result: dict, model_key: str, true_label: str | None = None) -> None:
    is_nli  = (model_key == "v3-nli")
    has_ec  = result["has_ec"]
    verdict = result["verdict"]
    v_icon  = VERDICT_META.get(verdict, ("?",))[0]
    bd      = result["evidence_breakdown"]

    if true_label is not None:
        t_icon = VERDICT_META.get(true_label, ("?",))[0]
        match  = verdict == true_label
        st.markdown(
            f"{'✅' if match else '❌'} &nbsp; "
            f"True: **{t_icon} {true_label}** &nbsp;·&nbsp; "
            f"Predicted: **{v_icon} {verdict}**",
            unsafe_allow_html=True,
        )
        st.divider()

    for i, ev in enumerate(bd):
        preview = ev["text_short"][:100] + ("…" if len(ev["text_short"]) > 100 else "")
        st.markdown(f"**Evidence {i + 1}**  ·  *{preview}*")
        rows: list[tuple[str, str]] = []
        if is_nli and ev.get("nli_probs"):
            nli = ev["nli_probs"]
            e, c, n = nli["entailment"], nli["contradiction"], nli["neutral"]
            rows.append(("① NLI cross-encoder",
                         f"entail `{e:.1%}` &nbsp; contra `{c:.1%}` &nbsp; neutral `{n:.1%}`"))
        stance = ev["stance"]
        s_icon = {"supports": "🟢", "refutes": "🔴", "neutral": "⚪"}.get(stance, "⚪")
        rows.append(("② Stance head (H1)", f"{s_icon} **{stance}** &nbsp; conf `{ev['stance_confidence']:.0%}`"))
        rows.append(("③ IS head (H2)", f"`{ev['is_score']:.3f}`"))
        if has_ec:
            rows.append(("④ EC formula",
                         f"`{ev['ec_score']:.3f}` &nbsp; ST `{ev['source_trust']:.2f}` · EW `{ev['evidence_weight']:.2f}` · IS `{ev['is_score']:.3f}`"))
        pram = PRAMANA_SHORT.get(ev.get("modality", "web_text"), "—")
        mod  = MODALITY_LABELS.get(ev.get("modality", "web_text"), ev.get("modality", ""))
        src  = SOURCE_LABELS.get(ev.get("source_type", "unknown"), ev.get("source_type", ""))
        rows.append(("⑤ Pramana", f"{pram} · {mod} · {src}"))
        st.markdown("| Layer | Value |\n|---|---|\n" + "".join(
            f"| {label} | {value} |\n" for label, value in rows))
        if i < len(bd) - 1:
            st.markdown("")

    st.divider()
    if has_ec:
        sup, ref = result["support_score"], result["refute_score"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**⑥ EC aggregation** &nbsp; support `{sup:.3f}` / refute `{ref:.3f}`")
            st.progress(min(sup, 1.0), text=f"🟢 support {sup:.3f}")
            st.progress(min(ref, 1.0), text=f"🔴 refute  {ref:.3f}")
        with c2:
            probs = result["verdict_probs"]
            st.markdown("**⑦ Verdict head**")
            for lbl, p in zip(VERDICT_LABELS, probs):
                st.progress(p, text=f"{VERDICT_META[lbl][0]} {p:.0%}")
    else:
        probs = result["verdict_probs"]
        st.markdown("**④ Verdict head**")
        for lbl, p in zip(VERDICT_LABELS, probs):
            st.progress(p, text=f"{VERDICT_META[lbl][0]} {p:.0%}")
    st.markdown(f"**Final verdict: {v_icon} {verdict.upper().replace('_', ' ')}**")


# ── Debug view ────────────────────────────────────────────────────────────────

def render_debug_view(result: dict, claim: str) -> None:
    bd     = result["evidence_breakdown"]
    is_nli = any(ev.get("nli_probs") is not None for ev in bd)
    has_ec = result["has_ec"]

    st.markdown("**Claim**")
    st.markdown(f"> {claim}")
    claim_triples = result.get("claim_triples") or []
    if claim_triples:
        st.markdown("*Claim triples:*")
        st.markdown("".join(render_triple(t) for t in claim_triples), unsafe_allow_html=True)

    st.markdown("**Evidence Items**")
    for i, ev in enumerate(bd):
        stance  = ev["stance"]
        s_cls   = STANCE_CHIP_CLS.get(stance, "chip-gray")
        with st.expander(f"Evidence {i + 1} · {stance} · {ev['text_short'][:60]}", expanded=False):
            st.markdown(ev["text"])
            triples = ev.get("triples") or []
            if triples:
                st.markdown("*Triples:*")
                st.markdown("".join(render_triple(t) for t in triples), unsafe_allow_html=True)
            st.divider()
            pram    = PRAMANA_SHORT.get(ev.get("modality", "web_text"), "Shabda")
            mod_lbl = MODALITY_LABELS.get(ev.get("modality", "web_text"), ev.get("modality", ""))
            src_lbl = SOURCE_LABELS.get(ev.get("source_type", "unknown"), ev.get("source_type", ""))
            ec_val  = ev.get("ec_score", 0.0)
            ec_cls  = "chip-green" if ec_val > 0.5 else "chip-amber" if ec_val > 0.2 else "chip-gray"
            prop_rows: list[tuple[str, str]] = [
                ("Pramana",         chip(pram, "chip-blue")),
                ("Modality",        chip(mod_lbl, "chip-gray")),
                ("Source type",     chip(src_lbl, "chip-gray")),
                ("Source trust",    f"`{ev.get('source_trust', 0):.3f}`"),
                ("Evidence weight", f"`{ev.get('evidence_weight', 0):.3f}`"),
                ("IS score",        f"`{ev.get('is_score', 0):.3f}`"),
            ]
            if has_ec:
                prop_rows.append(("EC score", chip(f"{ec_val:.3f}", ec_cls)))
            prop_rows.append(("Stance",            chip(stance, s_cls)))
            prop_rows.append(("Stance confidence", f"`{ev.get('stance_confidence', 0):.3f}`"))
            nli = ev.get("nli_probs")
            if is_nli and nli:
                prop_rows.extend([
                    ("NLI entailment",    chip(f"{nli['entailment']:.3f}",    "chip-green" if nli['entailment'] > 0.5 else "chip-gray")),
                    ("NLI contradiction", chip(f"{nli['contradiction']:.3f}", "chip-red" if nli['contradiction'] > 0.5 else "chip-gray")),
                    ("NLI neutral",       chip(f"{nli['neutral']:.3f}",       "chip-gray")),
                ])
            st.markdown("| Property | Value |\n|---|---|\n" + "".join(
                f"| {p} | {v} |\n" for p, v in prop_rows), unsafe_allow_html=True)

    st.markdown("**Verdict Summary**")
    render_verdict_card(result)
    if has_ec:
        sup = result.get("support_score", 0.0)
        ref = result.get("refute_score", 0.0)
        st.progress(min(sup, 1.0), text=f"🟢 support  {sup:.3f}")
        st.progress(min(ref, 1.0), text=f"🔴 refute   {ref:.3f}")


# ── Compare results (All Models) ──────────────────────────────────────────────

def render_compare_results(results: dict[str, dict | str]) -> None:
    cols = st.columns(4)
    for col, key in zip(cols, ["baseline", "v1-hgnn", "v2-hgnn", "v3-nli"]):
        result = results.get(key)
        with col:
            st.markdown(f"**{key}**")
            if result is None:
                st.info("—")
                continue
            if isinstance(result, str):
                st.error(result[:80])
                continue
            render_verdict_card(result)
            if result["has_ec"]:
                s, r = result["support_score"], result["refute_score"]
                st.caption(f"EC sup `{s:.3f}` ref `{r:.3f}`")
    st.divider()
    best = next(
        (k for k in ["v3-nli", "v2-hgnn"] if k in results and not isinstance(results[k], str)),
        None,
    )
    if best:
        st.markdown(f"*Layer trace — **{best}***")
        render_layerwise(results[best], best)


# ── Evidence card rendering ───────────────────────────────────────────────────

def render_evidence_cards() -> None:
    c_add, c_hint = st.columns([1, 5])
    with c_add:
        if st.button("＋ Add", use_container_width=True):
            st.session_state.evidence_list.append(blank_ev())
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
                _m = ev.get("modality", "web_text")
                mod = st.selectbox(
                    "Modality", MODALITIES, key=f"mod_{i}",
                    index=MODALITIES.index(_m if _m in MODALITIES else "web_text"),
                    format_func=lambda m: f"{PRAMANA_SHORT[m]} · {MODALITY_LABELS[m]}",
                )
                st.session_state.evidence_list[i]["modality"] = mod
            with c2:
                _s = ev.get("source_type", "unknown")
                src = st.selectbox(
                    "Source", SOURCE_TYPES, key=f"src_{i}",
                    index=SOURCE_TYPES.index(_s if _s in SOURCE_TYPES else "unknown"),
                    format_func=lambda s: SOURCE_LABELS[s],
                )
                st.session_state.evidence_list[i]["source_type"] = src


# ── PyVis interactive graph ───────────────────────────────────────────────────

# Verdict / stance int→label (mirroring src/model/data/types.py)
_INT_TO_VERDICT = {0: "supported", 1: "refuted", 2: "not_enough_evidence"}
_INT_TO_STANCE  = {0: "supports",  1: "refutes",  2: "neutral"}


def build_pyvis_html(
    hetero_data,                      # torch_geometric.data.HeteroData
    claim_text: str = "",
    ev_texts: list[str] | None = None,
    height: str = "540px",
) -> str:
    """Build an interactive pyvis HTML string from a HeteroData object.

    Node colours:
      CLAIM     — blue (#1d4ed8)
      supports  — green (#16a34a)
      refutes   — red  (#dc2626)
      neutral   — gray (#6b7280)
    Edge colours:
      has_evidence   CLAIM→EV  blue solid
      connected_to   EV→CLAIM  orange dashed
      co_evidence    EV↔EV     gray dashed
    """
    from pyvis.network import Network
    import torch

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#f8fafc",
        font_color="#1e293b",
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 140},
        "stabilization": {"iterations": 150}
      },
      "edges": {"smooth": {"type": "dynamic"}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    # ── Node type stores ──────────────────────────────────────────────────────
    try:
        from src.model.data.types import NodeType, EdgeType
    except ImportError:
        NodeType = type("NT", (), {"CLAIM": "claim", "EVIDENCE": "evidence"})()
        EdgeType = type("ET", (), {
            "HAS_EVIDENCE": "has_evidence",
            "CONNECTED_TO": "connected_to",
            "CO_EVIDENCE":  "co_evidence",
        })()

    claim_store = hetero_data[NodeType.CLAIM]
    ev_store    = hetero_data[NodeType.EVIDENCE]

    # ── CLAIM node ────────────────────────────────────────────────────────────
    verdict_y   = int(claim_store.y[0].item()) if hasattr(claim_store, "y") else -1
    verdict_lbl = _INT_TO_VERDICT.get(verdict_y, "?")
    claim_snip  = (claim_text[:60] + "…") if len(claim_text) > 60 else claim_text
    claim_title = (
        f"<b>CLAIM</b><br>"
        f"Verdict (GT): <b>{verdict_lbl}</b><br>"
        f"{claim_snip}<br>"
        f"x: [{claim_store.x.shape[1]} dims]"
    )
    net.add_node(
        "C0", label="CLAIM\n" + verdict_lbl,
        title=claim_title,
        color={"background": "#1d4ed8", "border": "#1e40af", "highlight": {"background": "#2563eb"}},
        font={"color": "white", "size": 13, "bold": True},
        size=40, shape="box",
    )

    # ── EVIDENCE nodes ────────────────────────────────────────────────────────
    n_ev = ev_store.x.shape[0]
    stance_y  = ev_store.stance_y.tolist() if hasattr(ev_store, "stance_y") else [2] * n_ev
    is_y      = ev_store.is_y.tolist()     if hasattr(ev_store, "is_y")     else [0.0] * n_ev
    ew_vals   = ev_store.ew.tolist()        if hasattr(ev_store, "ew")       else [0.0] * n_ev
    st_vals   = ev_store.st.tolist()        if hasattr(ev_store, "st")       else [0.0] * n_ev

    _ev_bg  = {0: "#d1fae5", 1: "#fee2e2", 2: "#f3f4f6"}
    _ev_brd = {0: "#16a34a", 1: "#dc2626", 2: "#9ca3af"}
    _ev_fc  = {0: "#14532d", 1: "#7f1d1d", 2: "#374151"}

    for i in range(n_ev):
        si        = int(stance_y[i])
        stance_lb = _INT_TO_STANCE.get(si, "?")
        text_snip = ""
        if ev_texts and i < len(ev_texts):
            t = ev_texts[i]
            text_snip = (t[:80] + "…") if len(t) > 80 else t
        ev_title = (
            f"<b>EV {i+1}</b><br>"
            f"Stance (GT): <b>{stance_lb}</b><br>"
            f"IS_y:  {is_y[i]:.4f}<br>"
            f"EW:    {ew_vals[i]:.4f}<br>"
            f"ST:    {st_vals[i]:.4f}<br>"
            f"x: [{ev_store.x.shape[1]} dims]"
        )
        if text_snip:
            ev_title += f"<br><i>{text_snip}</i>"

        net.add_node(
            f"E{i}",
            label=f"EV {i+1}\n{stance_lb}\nIS={is_y[i]:.3f}",
            title=ev_title,
            color={
                "background": _ev_bg.get(si, "#f3f4f6"),
                "border":     _ev_brd.get(si, "#9ca3af"),
                "highlight":  {"background": _ev_bg.get(si, "#f3f4f6")},
            },
            font={"color": _ev_fc.get(si, "#374151"), "size": 11},
            size=22, shape="box",
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    for etype, estore in hetero_data._edge_store_dict.items():
        ei = estore.get("edge_index")
        if ei is None:
            continue
        src_type, rel, dst_type = etype
        rel_name = rel.value if hasattr(rel, "value") else str(rel)

        if "has_evidence" in rel_name:
            color, dashes = "#1d4ed8", False
        elif "connected_to" in rel_name:
            color, dashes = "#f97316", True
        else:  # co_evidence
            color, dashes = "#9ca3af", True

        for s_idx, d_idx in ei.t().tolist():
            src_nt = src_type.value if hasattr(src_type, "value") else str(src_type)
            dst_nt = dst_type.value if hasattr(dst_type, "value") else str(dst_type)
            src_id = "C0"     if "claim"    in src_nt else f"E{s_idx}"
            dst_id = f"E{d_idx}" if "evidence" in dst_nt else "C0"
            net.add_edge(
                src_id, dst_id,
                title=rel_name,
                color={"color": color, "highlight": color},
                dashes=dashes,
                width=2 if not dashes else 1,
            )

    return net.generate_html()


# ── Model Computation Flow (DOT) ────────────────────────────────────────────────

def build_model_computation_dot(result: dict, model_name: str) -> str:
    """Computation-graph DOT showing the full GNN inference pipeline.

    Layers (top-to-bottom):
      L0  INPUT     — one column per evidence item + the claim
      L1  ENCODE    — text → hidden embedding (SentenceTransformer)
      L2  H1        — StanceHead → stance logits [supports/refutes/neutral]
                     (v3-nli: replaced by DeBERTa NLI for evidence side)
      L3  H2        — ISHead    → IS scalar [0, 1]
      L4  EC        — EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)
      L5  AGG       — product-of-complements aggregation per direction
      L6  DECISION  — symbolic threshold check (if model ≠ baseline)
      L7  VERDICT   — output
    """
    bd      = result["evidence_breakdown"]
    has_ec  = result["has_ec"]
    verdict = result["verdict"]
    sup     = result.get("support_score", 0.0)
    ref     = result.get("refute_score",  0.0)
    is_nli  = (model_name == "v3-nli")

    n = len(bd)

    v_fill  = {"supported": "#166534", "refuted": "#991b1b", "not_enough_evidence": "#92400e"}
    s_fill  = {"supports": "#d1fae5", "refutes": "#fee2e2", "neutral": "#f3f4f6"}
    s_fc    = {"supports": "#14532d", "refutes": "#7f1d1d", "neutral": "#374151"}

    def _esc(s: str) -> str:
        return s.replace('"', "'").replace("\n", " ").replace("\\", "/")

    lines = [
        "digraph GNN_Flow {",
        '  graph [bgcolor="#f8fafc", rankdir=TB, pad="0.5", nodesep=1.0, ranksep=0.7,',
        f'         label="Model Computation Flow — {model_name}", labelloc=t,',
        '         fontname="Arial", fontsize=11]',
        '  node  [fontname="Arial", fontsize=9, style=filled, margin="0.12,0.06"]',
        '  edge  [fontname="Arial", fontsize=8, color="#6b7280"]',
        "",
    ]

    # ── L0: INPUT ─────────────────────────────────────────────────────────────
    lines.append("  // ── L0: INPUTS ────────────────────────────────────────────────")
    lines.append("  { rank=same")
    for i, ev in enumerate(bd):
        text_s = _esc((ev.get("text_short") or ev.get("text", ""))[:40])
        src = ev.get("source_type", "?")
        pramana = ev.get("pramana", "—")
        st_v = ev.get("source_trust", 0.0)
        lines.append(
            f'    ev_in_{i} [label="EV {i+1} input\\n{text_s}\\n'
            f'ST={st_v:.3f}  src={src}\\nPramana={pramana}",'
            f' shape=rect, fillcolor="#f1f5f9", fontcolor="#1e293b"]'
        )
    lines.append("  }")
    lines.append("")

    # ── L1: ENCODE ────────────────────────────────────────────────────────────
    lines.append("  // ── L1: ENCODE ────────────────────────────────────────────────")
    enc_label = "DeBERTa NLI\\nencoder" if is_nli else "SentenceTransformer\\nencoder"
    lines.append("  { rank=same")
    for i in range(n):
        lines.append(
            f'    ev_enc_{i} [label="ENCODE\\n{enc_label}", shape=rect,'
            f' fillcolor="#dbeafe", fontcolor="#1e40af"]'
        )
    lines.append("  }")
    lines.append("")

    # ── L2: H1 STANCE ─────────────────────────────────────────────────────────
    h1_title = "H1: NLI → stance" if is_nli else "H1: StanceHead"
    lines.append(f"  // ── L2: {h1_title} ────────────────────────────────────")
    lines.append("  { rank=same")
    for i, ev in enumerate(bd):
        stance   = ev.get("stance", "neutral")
        s_conf   = ev.get("stance_confidence", 0.0)
        nli_p    = ev.get("nli_probs")
        if nli_p:
            lbl = (
                f"{h1_title}\\n"
                f"ent={nli_p['entailment']:.3f}\\n"
                f"con={nli_p['contradiction']:.3f}\\n"
                f"neu={nli_p['neutral']:.3f}\\n"
                f"→ {stance} ({s_conf:.3f})"
            )
        else:
            lbl = f"{h1_title}\\n→ {stance}\\nconf={s_conf:.3f}"
        lines.append(
            f'    ev_h1_{i} [label="{lbl}", shape=rect,'
            f' fillcolor="{s_fill.get(stance, "#f3f4f6")}",'
            f' fontcolor="{s_fc.get(stance, "#374151")}"]'
        )
    lines.append("  }")
    lines.append("")

    # ── L3: H2 IS HEAD ────────────────────────────────────────────────────────
    lines.append("  // ── L3: H2 ISHead ─────────────────────────────────────────────")
    lines.append("  { rank=same")
    for i, ev in enumerate(bd):
        is_v = ev.get("is_score", 0.0)
        ew_v = ev.get("evidence_weight", 0.0)
        lines.append(
            f'    ev_h2_{i} [label="H2: ISHead\\nIS={is_v:.4f}\\nEW={ew_v:.4f}", shape=rect,'
            f' fillcolor="#fef9c3", fontcolor="#713f12"]'
        )
    lines.append("  }")
    lines.append("")

    # ── L4: EC FORMULA (only when has_ec) ─────────────────────────────────────
    if has_ec:
        lines.append("  // ── L4: EC Formula ──────────────────────────────────────────────")
        lines.append("  { rank=same")
        for i, ev in enumerate(bd):
            is_v  = ev.get("is_score", 0.0)
            ew_v  = ev.get("evidence_weight", 0.0)
            st_v  = ev.get("source_trust", 0.0)
            ec_v  = ev.get("ec_score", 0.0)
            lines.append(
                f'    ev_ec_{i} [label="EC Formula\\n'
                f'1-(1-{st_v:.3f})^({ew_v:.3f}×{is_v:.3f})\\n'
                f'= EC_i = {ec_v:.4f}", shape=rect,'
                f' fillcolor="#f3e8ff", fontcolor="#6b21a8"]'
            )
        lines.append("  }")
        lines.append("")

    # ── L5: AGGREGATION ───────────────────────────────────────────────────────
    if has_ec:
        sup_evs = [ev.get("ec_score", 0.0) for ev in bd if ev.get("stance") == "supports"]
        ref_evs = [ev.get("ec_score", 0.0) for ev in bd if ev.get("stance") == "refutes"]
        lines.append("  // ── L5: Aggregation ─────────────────────────────────────────────")

        sup_terms = " × ".join(f"(1-{e:.3f})" for e in sup_evs) if sup_evs else "1.0"
        ref_terms = " × ".join(f"(1-{e:.3f})" for e in ref_evs) if ref_evs else "1.0"
        lines.append(
            f'  agg [label="EC Aggregation (product-of-complements)\\n'
            f'EC_sup = 1 - {sup_terms}\\n'
            f'       = {sup:.4f}\\n'
            f'EC_ref = 1 - {ref_terms}\\n'
            f'       = {ref:.4f}",'
            f' shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8", penwidth=2,'
            f' fontsize=10]'
        )
        lines.append("")

    # ── L6: DECISION ──────────────────────────────────────────────────────────
    if has_ec:
        thr = 0.35
        if sup > thr and ref > thr:
            dec_lbl = f"CONFLICTING\\nsup={sup:.3f} > {thr}  AND\\nref={ref:.3f} > {thr}\\n→ fall to VerdictHead"
            dec_fill, dec_fc = "#fef3c7", "#92400e"
        elif sup > thr:
            dec_lbl = f"SYMBOLIC OVERRIDE\\nsup={sup:.3f} > {thr}\\n→ SUPPORTED"
            dec_fill, dec_fc = "#d1fae5", "#14532d"
        elif ref > thr:
            dec_lbl = f"SYMBOLIC OVERRIDE\\nref={ref:.3f} > {thr}\\n→ REFUTED"
            dec_fill, dec_fc = "#fee2e2", "#7f1d1d"
        else:
            dec_lbl = f"EC WEAK\\nsup={sup:.3f}, ref={ref:.3f}\\nboth ≤ {thr}\\n→ fall to VerdictHead"
            dec_fill, dec_fc = "#f3f4f6", "#374151"
        lines.append(
            f'  decision [label="{dec_lbl}", shape=rect,'
            f' fillcolor="{dec_fill}", fontcolor="{dec_fc}", penwidth=2]'
        )
        lines.append("")

    # ── L7: VERDICT ───────────────────────────────────────────────────────────
    vp = result.get("verdict_probs") or [0.0, 0.0, 0.0]
    vp_labels = ["supported", "refuted", "not_enough_evidence"]
    vp_str = "  ".join(f"{lb[:3]}={p:.3f}" for lb, p in zip(vp_labels, vp))
    verdict_upper = verdict.replace("_", " ").upper()
    lines.append(
        f'  verdict [label="VerdictHead\\n{vp_str}\\n-> {verdict_upper}",'
        f' shape=rect, fillcolor="{v_fill.get(verdict, "#1d4ed8")}",'
        f' fontcolor="white", fontsize=11, penwidth=2]'
    )
    lines.append("")

    # ── Edges ──────────────────────────────────────────────────────────────────
    lines.append("  // ── Edges ──────────────────────────────────────────────────────────────")
    for i in range(n):
        lines.append(f"  ev_in_{i} -> ev_enc_{i}")
        lines.append(f"  ev_enc_{i} -> ev_h1_{i}")
        lines.append(f"  ev_enc_{i} -> ev_h2_{i}")
        if has_ec:
            lines.append(f"  ev_h1_{i} -> ev_ec_{i} [label=\"EW\"]")
            lines.append(f"  ev_h2_{i} -> ev_ec_{i} [label=\"IS\"]")
            stance = bd[i].get("stance", "neutral")
            if stance != "neutral":
                lines.append(f"  ev_ec_{i} -> agg [label=\"{stance}\", style=dashed]")
        else:
            lines.append(f"  ev_h1_{i} -> verdict [label=\"stance {i+1}\"]")
            lines.append(f"  ev_h2_{i} -> verdict [label=\"IS {i+1}\"]")
    if has_ec:
        lines.append("  agg -> decision")
        lines.append("  decision -> verdict")
    lines.append("")
    lines.append("}")
    return "\n".join(lines)


# ── Claim graph (DOT) ─────────────────────────────────────────────────────────

def build_claim_dot(claim: str, result: dict) -> str:
    """Build a Graphviz DOT string visualising the heterogeneous GNN graph.

    Node types:
      CLAIM   — single claim node (coloured by verdict)
      EV_i    — evidence nodes (coloured by stance)
      AGG     — EC aggregate diamond (only when EC is active)
    Edge semantics:
      CLAIM → EV_i   labelled with stance and EC score
      EV_i  → AGG    dashed, labelled with per-evidence EC contribution
    """
    bd      = result["evidence_breakdown"]
    has_ec  = result["has_ec"]
    verdict = result["verdict"]

    claim_short = claim[:55].replace('"', "'").replace("\n", " ")
    v_colors = {"supported": "#166534", "refuted": "#991b1b", "not_enough_evidence": "#92400e"}
    s_fill   = {"supports": "#d1fae5", "refutes": "#fee2e2", "neutral": "#f3f4f6"}
    s_fc     = {"supports": "#14532d", "refutes": "#7f1d1d", "neutral": "#374151"}
    s_edge   = {"supports": "#16a34a", "refutes": "#dc2626", "neutral": "#9ca3af"}

    lines = [
        "digraph GNN_Graph {",
        '  graph [bgcolor="#f8fafc", rankdir=TB, pad="0.5", nodesep=0.8, ranksep=0.7,'
        '         label="HeteroGraph — CLAIM · EVIDENCE · EC_AGG nodes", labelloc=t,'
        '         fontname="Arial", fontsize=10]',
        '  node  [fontname="Arial", fontsize=9, style=filled, margin="0.14,0.07"]',
        '  edge  [fontname="Arial", fontsize=8]',
        "",
        "  // ── CLAIM node (NodeType.CLAIM) ────────────────────────────────",
        f'  claim [label="[ CLAIM ]\\n{claim_short}\\n▸ {verdict.replace("_"," ").upper()}",'
        f'         shape=rect, fillcolor="{v_colors.get(verdict, "#1d4ed8")}", fontcolor="white",'
        f'         fontsize=10, penwidth=2]',
        "",
        "  // ── EVIDENCE nodes (NodeType.EVIDENCE) ─────────────────────────",
    ]

    for i, ev in enumerate(bd):
        stance   = ev.get("stance", "neutral")
        text_s   = (ev.get("text_short") or ev.get("text", ""))[:45].replace('"', "'").replace("\n", " ")
        is_v     = ev.get("is_score", 0.0)
        ec_v     = ev.get("ec_score", 0.0)
        st_v     = ev.get("source_trust", 0.0)
        ew_v     = ev.get("evidence_weight", 0.0)
        pramana  = ev.get("pramana", "—")
        src_type = ev.get("source_type", "?")

        # Node label
        lbl_lines = [f"[ EV {i+1} ]  {text_s}"]
        lbl_lines.append(f"stance: {stance}  IS: {is_v:.3f}  EW: {ew_v:.3f}")
        lbl_lines.append(f"ST: {st_v:.3f}  Pramana: {pramana}  ({src_type})")
        if has_ec:
            lbl_lines.append(f"EC_i = {ec_v:.4f}")
        node_lbl = "\\n".join(lbl_lines)

        # Edge from claim to evidence (shows how graph is structured)
        edge_lbl = f"{stance}"
        if has_ec:
            edge_lbl += f"\\nIS={is_v:.3f}"
        edge_color = s_edge.get(stance, "#9ca3af")

        lines += [
            f'  ev{i} [label="{node_lbl}", shape=rect,'
            f'         fillcolor="{s_fill.get(stance, "#f3f4f6")}",'
            f'         fontcolor="{s_fc.get(stance, "#374151")}"]',
            f'  claim -> ev{i} [label="{edge_lbl}", color="{edge_color}",'
            f'                  fontcolor="{edge_color}"]',
            "",
        ]

    # EC aggregate node
    if has_ec:
        sup, ref = result.get("support_score", 0.0), result.get("refute_score", 0.0)
        n_sup = sum(1 for ev in bd if ev.get("stance") == "supports")
        n_ref = sum(1 for ev in bd if ev.get("stance") == "refutes")
        lines += [
            "  // ── EC AGGREGATE node ──────────────────────────────────────────",
            f'  agg [label="[ EC AGGREGATE ]\\n'
            f'EC_support = {sup:.4f}  (from {n_sup} ev)\\n'
            f'EC_refute  = {ref:.4f}  (from {n_ref} ev)\\n'
            f'threshold = 0.35",',
            f'       shape=diamond, fillcolor="#eff6ff", fontcolor="#1d4ed8", penwidth=2]',
        ]
        for i, ev in enumerate(bd):
            ec_v   = ev.get("ec_score", 0.0)
            stance = ev.get("stance", "neutral")
            if stance != "neutral":
                ec_color = s_edge.get(stance, "#9ca3af")
                lines.append(
                    f'  ev{i} -> agg [style=dashed, label="EC={ec_v:.4f}",'
                    f' color="{ec_color}", fontcolor="{ec_color}"]'
                )
        lines.append("")

    lines.append("}")
    return "\n".join(lines)
