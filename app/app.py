"""Streamlit demo — Epistemic FactKG claim verifier."""

from __future__ import annotations

import io
import json
import random
from collections import Counter
from pathlib import Path

import streamlit as st

from predictor import EpistemicPredictor

# ── Constants ─────────────────────────────────────────────────────────────────

def _build_model_labels() -> dict[str, str]:
    _EVAL_ROOT = Path("out/reports/model")
    _META = [
        ("v3-nli",   "NLI + Hybrid  "),
        ("v2-hgnn",  "Hybrid        "),
        ("v1-hgnn",  "Pure Symbolic "),
        ("baseline", "No EC         "),
    ]
    result = {}
    for key, desc in _META:
        try:
            acc = json.loads((_EVAL_ROOT / key / "eval" / "verdict_metrics.json").read_text())["accuracy"]
            acc_str = f"{acc:.1%}"
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            acc_str = "—"
        result[key] = f"{key} — {desc}·  {acc_str}"
    return result


_MODELS = _build_model_labels()
_ALL_KEY = "all"

from _constants import (
    MODALITIES as _MODALITIES, MODALITY_LABELS as _MODALITY_LABELS,
    PRAMANA_SHORT as _PRAMANA_SHORT, SOURCE_LABELS as _SOURCE_LABELS,
    SOURCE_TYPES as _SOURCE_TYPES, VERDICT_META as _VERDICT_META,
    VERDICT_LABELS as _VERDICT_LABELS, VERDICT_CSS as _VERDICT_CSS,
    STANCE_CHIP_CLS as _STANCE_CHIP_CLS,
)

_DATA_JSONL = Path("out/data/training/epistemic_factkg_training.jsonl")
_TEST_IDX   = Path("out/data/splits/test_indices.json")

_MODEL_DESCRIPTIONS: dict[str, str] = {
    "baseline": "GNN encoder → VerdictHead only. No epistemic confidence formula.",
    "v1-hgnn":  "Adds EC formula. H1 stance + H2 IS feed a symbolic decision at 0.35.",
    "v2-hgnn":  "Hybrid: EC aggregation jointly feeds VerdictHead as extra input.",
    "v3-nli":   "v2-hgnn + frozen DeBERTa-v3-small NLI replaces H1 in the EC path.",
}

_CSS = """
<style>
:root {
  --bg-page:#fafafa; --bg-card:#ffffff; --border:#e5e7eb;
  --text-primary:#111827; --text-secondary:#6b7280; --text-muted:#9ca3af;
  --green:#15803d; --green-bg:#f0fdf4; --green-text:#166534;
  --red:#991b1b;   --red-bg:#fff1f2;   --red-text:#991b1b;
  --amber:#92400e; --amber-bg:#fffbeb; --amber-text:#92400e;
  --blue:#1d4ed8;  --blue-bg:#eff6ff;
  --purple:#6d28d9;--purple-bg:#f5f3ff;
  --layer-input:#6b7280; --layer-encoder:#1d4ed8; --layer-stance:#b45309;
  --layer-nli:#6d28d9;   --layer-ec:#15803d;      --layer-verdict:#991b1b;
  --radius:8px;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -1px rgba(0,0,0,.04);
}
.arch-box {
  background:var(--bg-card); border:1px solid var(--border);
  border-left:4px solid var(--layer-input); border-radius:var(--radius);
  padding:10px 14px; margin:4px 0;
  font-family:'JetBrains Mono',ui-monospace,monospace; font-size:0.82rem;
  box-shadow:var(--shadow);
}
.arch-box.enc     { border-left-color:var(--layer-encoder); }
.arch-box.stance  { border-left-color:var(--layer-stance);  }
.arch-box.nli     { border-left-color:var(--layer-nli);     }
.arch-box.ec      { border-left-color:var(--layer-ec);      }
.arch-box.verdict { border-left-color:var(--layer-verdict); }
.arch-box-title {
  font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--text-muted); margin-bottom:6px;
}
.chip {
  display:inline-block; padding:1px 8px; border-radius:10px;
  font-size:0.72rem; font-weight:600; margin:1px; line-height:1.6;
}
.chip-green  { background:var(--green-bg);  color:var(--green-text); }
.chip-red    { background:var(--red-bg);    color:var(--red-text);   }
.chip-amber  { background:var(--amber-bg);  color:var(--amber-text); }
.chip-gray   { background:#f3f4f6;          color:#374151;           }
.chip-blue   { background:var(--blue-bg);   color:var(--blue);       }
.chip-purple { background:var(--purple-bg); color:var(--purple);     }
.verdict-card {
  border:1.5px solid var(--border); border-radius:var(--radius);
  padding:18px 20px; text-align:center; margin:8px 0;
  box-shadow:var(--shadow-md);
}
.verdict-card.sup { background:var(--green-bg); border-color:var(--green); color:var(--green-text); }
.verdict-card.ref { background:var(--red-bg);   border-color:var(--red);   color:var(--red-text);   }
.verdict-card.nei { background:var(--amber-bg); border-color:var(--amber); color:var(--amber-text); }
.verdict-label { font-size:1.45rem; font-weight:700; letter-spacing:-0.01em; }
.verdict-conf  { font-size:0.82rem; margin-top:4px; opacity:0.85; }
.minibar-wrap  { display:flex; border-radius:4px; overflow:hidden; height:6px; }
.minibar-seg   { height:100%; }
.triple-row    { display:flex; align-items:center; gap:6px; margin:3px 0; flex-wrap:wrap; }
.stProgress > div > div > div { border-radius:3px; }
.decision-path {
  border-radius:6px; padding:8px 12px;
  font-size:0.82rem; font-weight:500; margin:6px 0;
  border-left:3px solid; line-height:1.5;
}
.dp-sup      { background:var(--green-bg); border-color:var(--green); color:var(--green-text); }
.dp-ref      { background:var(--red-bg);   border-color:var(--red);   color:var(--red-text);   }
.dp-conflict { background:var(--amber-bg); border-color:var(--amber); color:var(--amber-text); }
.dp-weak     { background:#f9fafb; border-color:var(--text-muted);   color:var(--text-secondary); }
.dp-baseline { background:#f9fafb; border-color:var(--layer-verdict); color:var(--layer-verdict); }
.model-desc {
  font-size:0.78rem; color:var(--text-secondary); background:#f9fafb;
  border-radius:6px; padding:7px 10px; margin:6px 0 0;
  line-height:1.5; border:1px solid var(--border);
}
.page-header { display:flex; align-items:baseline; gap:10px; margin-bottom:2px; }
.page-title  { font-size:1.55rem; font-weight:700; color:var(--text-primary); letter-spacing:-0.02em; margin:0; }
.page-badge  { font-size:0.72rem; font-weight:600; padding:2px 8px; border-radius:10px; background:var(--blue-bg); color:var(--blue); }
</style>
"""


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_test_records() -> list[dict]:
    if not _DATA_JSONL.exists() or not _TEST_IDX.exists():
        return []
    with open(_TEST_IDX, encoding="utf-8") as f:
        indices = set(json.load(f)["indices"])
    records = []
    with open(_DATA_JSONL, encoding="utf-8") as f:
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


# ── Model selector helper ────────────────────────────────────────────────────

def _model_selector(widget_key: str, allow_all: bool = False) -> str:
    """Compact horizontal model radio; returns the selected model key."""
    keys   = list(_MODELS.keys())
    labels = list(_MODELS.values())
    if allow_all:
        keys.append(_ALL_KEY)
        labels.append("All Models")
    idx = st.radio(
        "Model",
        range(len(keys)),
        format_func=lambda i: labels[i],
        horizontal=True,
        key=widget_key,
        label_visibility="collapsed",
    )
    selected = keys[idx]
    desc = _MODEL_DESCRIPTIONS.get(selected, "")
    if desc:
        st.caption(desc)
    return selected


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "evidence_list":      [_blank_ev()],
        "last_claim":         "",
        "_random_true_label": None,
        "eval_rows":          None,
        "inspect_idx":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Consume the pending claim from a Random button click.
    # We write to "claim_input" here, before the widget is instantiated, which
    # is the only point where Streamlit allows direct session_state assignment.
    pending = st.session_state.pop("_pending_claim", None)
    if pending is not None:
        st.session_state["claim_input"] = pending


def _blank_ev() -> dict:
    return {"text": "", "source_type": "unknown", "modality": "web_text"}


@st.cache_data(show_spinner=False)
def _load_all_records_indexed() -> dict[str, dict]:
    """All JSONL records keyed by their `id` field (row index as fallback)."""
    if not _DATA_JSONL.exists():
        return {}
    result: dict[str, dict] = {}
    with open(_DATA_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                result[r.get("id", str(i))] = r
    return result


def _load_record_into_state(rec: dict) -> None:
    """Push a record's claim + evidence into session state for the Verify tab."""
    old_n = len(st.session_state.get("evidence_list", []))
    for i in range(old_n + 6):
        for pfx in ("ev_", "mod_", "src_"):
            st.session_state.pop(f"{pfx}{i}", None)
    new_evs = [
        {
            "text":        ev.get("text", ""),
            "source_type": "academic" if "wikipedia" in ev.get("source_id", "") else "unknown",
            "modality":    ev.get("modality", "web_text"),
        }
        for ev in rec.get("evidence", [])[:4]
    ] or [_blank_ev()]
    # Use a staging key — writing directly to "claim_input" raises
    # StreamlitAPIException because the widget is already instantiated.
    # The pending value is consumed before the next render (see text_area below).
    st.session_state["_pending_claim"]     = rec["claim"]
    st.session_state["last_claim"]         = rec["claim"]
    st.session_state["evidence_list"]      = new_evs
    st.session_state["_random_true_label"] = rec.get("verdict", {}).get("label")
    for i, ev in enumerate(new_evs):
        st.session_state[f"ev_{i}"]  = ev["text"]
        st.session_state[f"mod_{i}"] = ev["modality"]
        st.session_state[f"src_{i}"] = ev["source_type"]


def _load_random_example() -> None:
    records = _load_test_records()
    if not records:
        st.warning("Test data not found.")
        return
    _load_record_into_state(random.choice(records))


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
                _m = ev.get("modality", "web_text")
                mod = st.selectbox(
                    "Modality", _MODALITIES, key=f"mod_{i}",
                    index=_MODALITIES.index(_m if _m in _MODALITIES else "web_text"),
                    format_func=lambda m: f"{_PRAMANA_SHORT[m]} · {_MODALITY_LABELS[m]}",
                )
                st.session_state.evidence_list[i]["modality"] = mod
            with c2:
                _s = ev.get("source_type", "unknown")
                src = st.selectbox(
                    "Source", _SOURCE_TYPES, key=f"src_{i}",
                    index=_SOURCE_TYPES.index(_s if _s in _SOURCE_TYPES else "unknown"),
                    format_func=lambda s: _SOURCE_LABELS[s],
                )
                st.session_state.evidence_list[i]["source_type"] = src


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _render_decision_path(result: dict) -> None:
    """Callout showing which symbolic decision path produced this verdict."""
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


def _render_arch_pipeline_bar(model_key: str) -> None:
    """Horizontal coloured pipeline strip summarising the model stages."""
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


def _chip(text: str, cls: str = "chip-gray") -> str:
    return f'<span class="chip {cls}">{text}</span>'


def _arch_box(css_class: str, step: str, title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="arch-box {css_class}">'
        f'<div class="arch-box-title">{step} &nbsp; {title}</div>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def _arch_arrow(label: str = "") -> None:
    if label:
        inner = (
            f"↓ &thinsp; <em style='color:var(--text-muted);font-size:0.75rem'>{label}</em>"
        )
    else:
        inner = "↓"
    st.markdown(
        f'<div style="text-align:center;color:var(--text-muted);padding:4px 0">{inner}</div>',
        unsafe_allow_html=True,
    )


def _triple_label(uri: str) -> str:
    seg = uri.split("/")[-1]
    return seg.split("|")[0] or seg


def _render_triple(triple: list) -> str:
    if not triple or len(triple) < 3:
        return ""
    s = _triple_label(str(triple[0]))
    p = str(triple[1])
    o = _triple_label(str(triple[2]))
    arrow = '<span style="color:var(--text-muted);font-size:0.75rem">→</span>'
    return (
        f'<div class="triple-row">'
        f'{_chip(s, "chip-blue")}{arrow}'
        f'{_chip(p, "chip-purple")}{arrow}'
        f'{_chip(o, "chip-green")}'
        f"</div>"
    )


def _render_verdict_card(result: dict) -> None:
    verdict = result["verdict"]
    probs   = result["verdict_probs"]
    icon, label, _ = _VERDICT_META.get(verdict, ("?", verdict.upper(), "#888"))
    css_cls = _VERDICT_CSS.get(verdict, "nei")
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

def _ev_table(headers: list[str], rows_data: list[list[str]]) -> str:
    th = "".join(
        f'<th style="padding:2px 5px;text-align:{"left" if j == 0 else "center"}">{h}</th>'
        for j, h in enumerate(headers)
    )
    trs = ""
    for row in rows_data:
        tds = "".join(
            f'<td style="padding:2px 5px;text-align:{"left" if j == 0 else "center"}">{cell}</td>'
            for j, cell in enumerate(row)
        )
        trs += f"<tr>{tds}</tr>"
    return (
        '<table style="font-size:0.75rem;border-collapse:collapse;width:100%">'
        f"<tr>{th}</tr>{trs}</table>"
    )


def _render_arch_flow(result: dict, model_key: str) -> None:
    _render_arch_pipeline_bar(model_key)
    is_nli    = model_key == "v3-nli"
    has_ec    = result["has_ec"]
    breakdown = result["evidence_breakdown"]
    n_ev      = len(breakdown)
    ev_dim    = "403d (400 + NLI 3d)" if is_nli else "400d"

    # ── ① INPUT ──────────────────────────────────────────────────────────────
    input_html = f"<div>Claim <b>390d</b> &nbsp;·&nbsp; Evidence &times;{n_ev} <b>{ev_dim}</b></div>"
    if is_nli:
        nli_rows = []
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs") or {}
            e = nli.get("entailment", 0)
            c = nli.get("contradiction", 0)
            n = nli.get("neutral", 0)
            nli_rows.append([
                f"ev{i + 1}",
                _chip(f"{e:.3f}", "chip-green" if e > 0.5 else "chip-gray"),
                _chip(f"{c:.3f}", "chip-red" if c > 0.5 else "chip-gray"),
                _chip(f"{n:.3f}", "chip-gray"),
            ])
        input_html += _ev_table(["ev", "entail", "contra", "neutral"], nli_rows)
    _arch_box("", "①", "INPUT FEATURES", input_html)

    _arch_arrow("HeteroConv · GAT 4 heads · 2 layers")

    # ── ② ENCODER ────────────────────────────────────────────────────────────
    _arch_box(
        "enc", "②", "GNN ENCODER",
        f"<div>claim_emb <b>256d</b> &nbsp;&middot;&nbsp; ev_emb &times;{n_ev} <b>256d</b></div>"
        '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">'
        "Claim context flows into ev_emb via has_evidence / connected_to edges</div>",
    )

    _arch_arrow("parallel heads")

    # ── ③ H1 ‖ H2 ─────────────────────────────────────────────────────────────
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        h1_rows = []
        for i, ev in enumerate(breakdown):
            stance = ev["stance"]
            conf   = ev["stance_confidence"]
            h1_rows.append([
                f"ev{i + 1}",
                _chip(stance, _STANCE_CHIP_CLS.get(stance, "chip-gray")),
                _chip(f"{conf:.0%}", "chip-gray"),
            ])
        _arch_box("stance", "③a", "STANCE HEAD (H1) &nbsp; Linear(256→3)",
                  _ev_table(["ev", "stance", "conf"], h1_rows))

    with c_h2:
        h2_rows = []
        for i, ev in enumerate(breakdown):
            is_val = ev["is_score"]
            is_cls = "chip-green" if is_val > 0.5 else "chip-amber" if is_val > 0.3 else "chip-gray"
            h2_rows.append([f"ev{i + 1}", _chip(f"{is_val:.3f}", is_cls)])
        _arch_box("stance", "③b", "IS HEAD (H2) &nbsp; Linear(256→1)",
                  _ev_table(["ev", "IS score"], h2_rows))

    # ── ④ NLI bypass (v3-nli only) ───────────────────────────────────────────
    if is_nli:
        _arch_arrow("v3-nli: NLI bypasses H1 in EC formula")
        nli_rows4 = []
        for i, ev in enumerate(breakdown):
            nli = ev.get("nli_probs") or {}
            sup_p = nli.get("entailment", 0)
            ref_p = nli.get("contradiction", 0)
            neu_p = nli.get("neutral", 0)
            best_lbl, _ = max(
                [("sup", sup_p), ("ref", ref_p), ("neu", neu_p)], key=lambda x: x[1]
            )
            ec_cls = {"sup": "chip-green", "ref": "chip-red", "neu": "chip-gray"}.get(best_lbl, "chip-gray")
            nli_rows4.append([
                f"ev{i + 1}",
                _chip(f"{sup_p:.3f}", "chip-green" if sup_p > 0.5 else "chip-gray"),
                _chip(f"{ref_p:.3f}", "chip-red" if ref_p > 0.5 else "chip-gray"),
                _chip(f"{neu_p:.3f}", "chip-gray"),
                _chip(best_lbl, ec_cls),
            ])
        nli_note = (
            '<div style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">'
            "Reorder: [contra, entail, neutral] → [ref, sup, neutral] for EC formula</div>"
        )
        _arch_box(
            "nli", "④",
            "NLI CROSS-ENCODER BYPASS &nbsp; (frozen DeBERTa-v3-small)",
            _ev_table(["ev", "entail→sup", "contra→ref", "neutral", "EC stance"], nli_rows4)
            + nli_note,
        )

    # ── ⑤ EC FORMULA ─────────────────────────────────────────────────────────
    if has_ec:
        _arch_arrow()
        ec_rows = []
        for i, ev in enumerate(breakdown):
            ec_val = ev["ec_score"]
            st_val = ev["source_trust"]
            ew_val = ev["evidence_weight"]
            is_val = ev["is_score"]
            ec_cls = "chip-green" if ec_val > 0.5 else "chip-amber" if ec_val > 0.2 else "chip-gray"
            ec_rows.append([
                f"ev{i + 1}",
                _chip(f"{st_val:.2f}", "chip-gray"),
                _chip(f"{ew_val:.2f}", "chip-gray"),
                _chip(f"{is_val:.3f}", "chip-gray"),
                _chip(f"{ec_val:.3f}", ec_cls),
            ])
        _arch_box("ec", "⑤", "EC FORMULA &nbsp; EC = 1−(1−ST)^(EW×IS)",
                  _ev_table(["ev", "ST", "EW", "IS", "EC"], ec_rows))

    # ── ⑥ EC AGGREGATION ─────────────────────────────────────────────────────
    if has_ec:
        _arch_arrow("1 − ∏(1 − ECᵢ × p_stanceᵢ)  across all evidence")
        sup = result["support_score"]
        ref = result["refute_score"]
        sup_w = min(sup * 100, 100)
        ref_w = min(ref * 100, 100)
        sup_chip = _chip(f"support {sup:.3f}", "chip-green" if sup > 0.35 else "chip-gray")
        ref_chip = _chip(f"refute {ref:.3f}", "chip-red" if ref > 0.35 else "chip-gray")

        def _bar(color: str, width: float) -> str:
            return (
                f'<div style="background:#e9ecef;border-radius:3px;height:6px;margin:3px 0">'
                f'<div style="background:var(--{color});height:100%;width:{width:.1f}%;border-radius:3px"></div>'
                f"</div>"
            )

        bars = (
            f'<div style="margin:4px 0">{sup_chip}{_bar("green", sup_w)}</div>'
            f'<div style="margin:4px 0">{ref_chip}{_bar("red", ref_w)}</div>'
        )
        _arch_box("ec", "⑥", "EC AGGREGATION", bars)

    # ── ⑦ VERDICT HEAD ───────────────────────────────────────────────────────
    vh_num = "⑦" if has_ec else "④"
    _arch_arrow()
    if has_ec:
        head_html = (
            "<div>cat([EC scores <b>2d</b>, claim_emb <b>256d</b>]) → <b>258d</b>"
            " → Linear(258→128) → ReLU → Dropout → Linear(128→3)</div>"
        )
    else:
        head_html = "<div>Linear(256→128) → ReLU → Dropout → Linear(128→3)</div>"

    probs = result["verdict_probs"]
    prob_colors = {"supported": "green", "refuted": "red", "not_enough_evidence": "amber"}
    prob_chip_cls = {"supported": "chip-green", "refuted": "chip-red", "not_enough_evidence": "chip-amber"}
    short_labels  = {"supported": "supported", "refuted": "refuted", "not_enough_evidence": "NEI"}

    prob_rows_html = ""
    for lbl, p in zip(_VERDICT_LABELS, probs):
        p_w    = p * 100
        color  = prob_colors[lbl]
        cls    = prob_chip_cls[lbl]
        slbl   = short_labels[lbl]
        prob_rows_html += (
            f'<div style="margin:4px 0;display:flex;align-items:center;gap:6px">'
            f'{_chip(slbl, cls)}'
            f'<div style="flex:1;background:#e9ecef;border-radius:3px;height:6px">'
            f'<div style="background:var(--{color});height:100%;width:{p_w:.1f}%;border-radius:3px"></div>'
            f"</div>"
            f'<span style="font-size:0.75rem;min-width:38px">{p:.1%}</span>'
            f"</div>"
        )
    _arch_box("verdict", vh_num, "VERDICT HEAD", head_html + prob_rows_html)

    # ── Final verdict ─────────────────────────────────────────────────────────
    _arch_arrow()
    _render_verdict_card(result)


# ── Layer-wise reasoning display ──────────────────────────────────────────────

def _render_layerwise(result: dict, model_key: str, true_label: str | None = None) -> None:
    is_nli    = (model_key == "v3-nli")
    has_ec    = result["has_ec"]
    verdict   = result["verdict"]
    v_icon    = _VERDICT_META.get(verdict, ("?",))[0]
    breakdown = result["evidence_breakdown"]

    if true_label is not None:
        t_icon = _VERDICT_META.get(true_label, ("?",))[0]
        match  = verdict == true_label
        st.markdown(
            f"{'✅' if match else '❌'} &nbsp; "
            f"True: **{t_icon} {true_label}** &nbsp;·&nbsp; "
            f"Predicted: **{v_icon} {verdict}**",
            unsafe_allow_html=True,
        )
        st.divider()

    for i, ev in enumerate(breakdown):
        preview = ev["text_short"][:100] + ("…" if len(ev["text_short"]) > 100 else "")
        st.markdown(f"**Evidence {i + 1}**  ·  *{preview}*")

        rows: list[tuple[str, str]] = []

        if is_nli and ev.get("nli_probs"):
            nli = ev["nli_probs"]
            e, c, n = nli["entailment"], nli["contradiction"], nli["neutral"]
            rows.append((
                "① NLI cross-encoder",
                f"entail `{e:.1%}` &nbsp; contra `{c:.1%}` &nbsp; neutral `{n:.1%}`",
            ))

        stance = ev["stance"]
        s_icon = {"supports": "🟢", "refutes": "🔴", "neutral": "⚪"}.get(stance, "⚪")
        rows.append((
            "② Stance head (H1)",
            f"{s_icon} **{stance}** &nbsp; conf `{ev['stance_confidence']:.0%}`",
        ))
        rows.append(("③ IS head (H2)", f"`{ev['is_score']:.3f}`"))

        if has_ec:
            ec = ev["ec_score"]
            st_v = ev["source_trust"]
            ew   = ev["evidence_weight"]
            is_v = ev["is_score"]
            rows.append((
                "④ EC formula",
                f"`{ec:.3f}` &nbsp; ST `{st_v:.2f}` · EW `{ew:.2f}` · IS `{is_v:.3f}`",
            ))

        pram = _PRAMANA_SHORT.get(ev.get("modality", "web_text"), "—")
        mod  = _MODALITY_LABELS.get(ev.get("modality", "web_text"), ev.get("modality", ""))
        src  = _SOURCE_LABELS.get(ev.get("source_type", "unknown"), ev.get("source_type", ""))
        rows.append(("⑤ Pramana", f"{pram} · {mod} · {src}"))

        tbl = "| Layer | Value |\n|---|---|\n" + "".join(
            f"| {label} | {value} |\n" for label, value in rows
        )
        st.markdown(tbl)

        if i < len(breakdown) - 1:
            st.markdown("")

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
        st.markdown("**④ Verdict head**")
        for lbl, p in zip(_VERDICT_LABELS, probs):
            icon = _VERDICT_META[lbl][0]
            st.progress(p, text=f"{icon} {p:.0%}")

    st.markdown(f"**Final verdict: {v_icon} {verdict.upper().replace('_', ' ')}**")


# ── Debug view ────────────────────────────────────────────────────────────────

def _render_debug_view(result: dict, claim: str) -> None:
    breakdown = result["evidence_breakdown"]
    is_nli    = any(ev.get("nli_probs") is not None for ev in breakdown)
    has_ec    = result["has_ec"]

    st.markdown("**Claim**")
    st.markdown(f"> {claim}")

    claim_triples = result.get("claim_triples") or []
    if claim_triples:
        st.markdown("*Claim triples:*")
        st.markdown(
            "".join(_render_triple(t) for t in claim_triples),
            unsafe_allow_html=True,
        )

    st.markdown("**Evidence Items**")
    for i, ev in enumerate(breakdown):
        stance   = ev["stance"]
        s_cls    = _STANCE_CHIP_CLS.get(stance, "chip-gray")
        label_tx = ev["text_short"][:60]

        with st.expander(f"Evidence {i + 1} · {stance} · {label_tx}", expanded=False):
            st.markdown(ev["text"])
            triples = ev.get("triples") or []
            if triples:
                st.markdown("*Triples:*")
                st.markdown(
                    "".join(_render_triple(t) for t in triples),
                    unsafe_allow_html=True,
                )
            st.divider()

            pram    = _PRAMANA_SHORT.get(ev.get("modality", "web_text"), "Shabda")
            mod_lbl = _MODALITY_LABELS.get(ev.get("modality", "web_text"), ev.get("modality", ""))
            src_lbl = _SOURCE_LABELS.get(ev.get("source_type", "unknown"), ev.get("source_type", ""))
            ec_val  = ev.get("ec_score", 0.0)
            ec_cls  = "chip-green" if ec_val > 0.5 else "chip-amber" if ec_val > 0.2 else "chip-gray"

            prop_rows: list[tuple[str, str]] = [
                ("Pramana",          _chip(pram, "chip-blue")),
                ("Modality",         _chip(mod_lbl, "chip-gray")),
                ("Source type",      _chip(src_lbl, "chip-gray")),
                ("Source trust",     f"`{ev.get('source_trust', 0):.3f}`"),
                ("Evidence weight",  f"`{ev.get('evidence_weight', 0):.3f}`"),
                ("IS score",         f"`{ev.get('is_score', 0):.3f}`"),
            ]
            if has_ec:
                prop_rows.append(("EC score", _chip(f"{ec_val:.3f}", ec_cls)))
            prop_rows.append(("Stance",            _chip(stance, s_cls)))
            prop_rows.append(("Stance confidence", f"`{ev.get('stance_confidence', 0):.3f}`"))

            nli = ev.get("nli_probs")
            if is_nli and nli:
                e_val = nli["entailment"]
                c_val = nli["contradiction"]
                n_val = nli["neutral"]
                prop_rows.extend([
                    ("NLI entailment",    _chip(f"{e_val:.3f}", "chip-green" if e_val > 0.5 else "chip-gray")),
                    ("NLI contradiction", _chip(f"{c_val:.3f}", "chip-red" if c_val > 0.5 else "chip-gray")),
                    ("NLI neutral",       _chip(f"{n_val:.3f}", "chip-gray")),
                ])

            tbl = "| Property | Value |\n|---|---|\n" + "".join(
                f"| {prop} | {val} |\n" for prop, val in prop_rows
            )
            st.markdown(tbl, unsafe_allow_html=True)

    st.markdown("**Verdict Summary**")
    _render_verdict_card(result)

    if has_ec:
        sup = result.get("support_score", 0.0)
        ref = result.get("refute_score", 0.0)
        st.progress(min(sup, 1.0), text=f"🟢 support  {sup:.3f}")
        st.progress(min(ref, 1.0), text=f"🔴 refute   {ref:.3f}")


# ── Compare results (All Models) ──────────────────────────────────────────────

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
                st.error(result[:80])
                continue
            _render_verdict_card(result)
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
        _render_layerwise(results[best], best)


# ── Evaluation export helpers ─────────────────────────────────────────────────

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
    sup = result.get("support_score", 0.0)
    ref = result.get("refute_score", 0.0)
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
    export = []
    for row in rows:
        result = row.get("result") or {}
        probs  = result.get("verdict_probs") or [0.0, 0.0, 0.0]
        sup    = result.get("support_score", 0.0)
        ref    = result.get("refute_score", 0.0)
        bd     = result.get("evidence_breakdown") or []
        pred   = row.get("pred")
        true   = row.get("true", "")

        has_placeholder = any(
            (ev.get("text") or "").strip().lower() in _PLACEHOLDER_TEXTS for ev in bd
        )
        has_qa = any(
            (ev.get("text") or "").lower().lstrip().startswith(_QA_PREFIXES) for ev in bd
        )
        ec_disagrees = (ref > 0.35 and pred != "refuted") or (sup > 0.35 and pred != "supported")

        # Evidence summary: keep full breakdown for diagnosis
        evidence_summary = [
            {
                "text":               ev.get("text", ""),
                "stance":             ev.get("stance"),
                "stance_confidence":  ev.get("stance_confidence"),
                "is_score":           ev.get("is_score"),
                "source_trust":       ev.get("source_trust"),
                "evidence_weight":    ev.get("evidence_weight"),
                "ec_score":           ev.get("ec_score"),
                "pramana":            _PRAMANA_SHORT.get(ev.get("modality", "web_text"), "—"),
                "source_type":        ev.get("source_type"),
                "nli_probs":          ev.get("nli_probs"),
            }
            for ev in bd
        ]

        export.append({
            "model":             row.get("model", ""),
            "source_dataset":    row.get("source", ""),
            "claim":             row.get("claim", ""),
            "true_label":        true,
            "predicted_label":   pred,
            "correct":           pred == true if pred is not None else None,
            "support_score":     round(sup, 4),
            "refute_score":      round(ref, 4),
            "verdict_probs":     {"supported": round(probs[0], 4), "refuted": round(probs[1], 4), "nei": round(probs[2], 4)},
            "has_placeholder_ev": has_placeholder,
            "has_qa_ev":         has_qa,
            "ec_disagrees":      ec_disagrees,
            "failure_pattern":   _failure_pattern(row),
            "evidence":          evidence_summary,
        })
    return json.dumps(export, indent=2, ensure_ascii=False)


# ── Data tab ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_dataset_stats() -> dict | None:
    if not _DATA_JSONL.exists():
        return None
    records = [
        json.loads(l) for l in _DATA_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    from collections import Counter as _Counter
    verdicts   = _Counter(r.get("verdict", {}).get("label", "unknown") for r in records)
    sources    = _Counter(r.get("provenance", {}).get("dataset", "unknown") for r in records)
    modalities = _Counter()
    for r in records:
        for ev in r.get("evidence", []):
            modalities[ev.get("modality", "unknown")] += 1
    splits: dict[str, int] = {}
    for split in ("train", "val", "test"):
        p = Path("out/data/splits") / f"{split}_indices.json"
        if p.exists():
            try:
                splits[split] = len(json.loads(p.read_text(encoding="utf-8"))["indices"])
            except Exception:
                splits[split] = 0
    return {
        "total":      len(records),
        "verdicts":   dict(verdicts),
        "sources":    dict(sources),
        "modalities": dict(modalities),
        "splits":     splits,
    }


def _render_data_tab() -> None:
    import pandas as pd
    stats = _load_dataset_stats()
    if stats is None:
        st.info(f"Training JSONL not found: `{_DATA_JSONL}`")
        return

    sp = stats["splits"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", stats["total"])
    c2.metric("Train",         sp.get("train", "—"))
    c3.metric("Val",           sp.get("val",   "—"))
    c4.metric("Test",          sp.get("test",  "—"))

    st.markdown("---")
    c_v, c_s = st.columns(2)
    with c_v:
        st.markdown("#### Verdict Distribution")
        st.bar_chart(pd.DataFrame.from_dict(stats["verdicts"], orient="index", columns=["count"]))
    with c_s:
        st.markdown("#### Source Dataset")
        st.bar_chart(pd.DataFrame.from_dict(stats["sources"], orient="index", columns=["count"]))

    st.markdown("#### Evidence Modality Distribution")
    st.bar_chart(pd.DataFrame.from_dict(stats["modalities"], orient="index", columns=["count"]))


# ── Reference tab ─────────────────────────────────────────────────────────────

def _render_reference_tab() -> None:
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("### EC Formula")
        st.code("EC_i = 1 − (1 − ST_i)^(EW_i × IS_i)", language=None)
        st.markdown(
            "| Symbol | Meaning | Range |\n|--------|---------|-------|\n"
            "| ST | Source Trust | [0, 1] |\n"
            "| EW | Evidence Weight (stance prob.) | [0, 1] |\n"
            "| IS | Inference Strength | [0, 1] |\n"
            "| EC | Epistemic Confidence | [0, 1] |"
        )

        st.markdown("### Decision Logic  `threshold = 0.35`")
        st.markdown(
            "| Condition | Outcome |\n|-----------|---------|\n"
            "| sup > 0.35 **and** ref > 0.35 | Conflicting → VerdictHead |\n"
            "| sup > 0.35 only | Symbolic override → **SUPPORTED** |\n"
            "| ref > 0.35 only | Symbolic override → **REFUTED** |\n"
            "| neither > 0.35 | EC weak → VerdictHead |\n"
            "| baseline model | VerdictHead always |"
        )

        st.markdown("### Source Trust Scale")
        st.markdown(
            "| Source Type | Default ST |\n|-------------|------------|\n"
            "| academic | 0.95 |\n"
            "| government | 0.90 |\n"
            "| news | 0.75 |\n"
            "| unknown | 0.60 |\n"
            "| social_media | 0.40 |"
        )

    with c_r:
        st.markdown("### Pramana (Epistemic Modalities)")
        st.markdown(
            "| Modality | Pramana | Meaning |\n|----------|---------|----------|\n"
            "| Web Text / PDF | Shabda | Testimony |\n"
            "| Image / Video / Audio | Pratyaksha | Perception |\n"
            "| Web Table | Upamana | Comparison |"
        )

        st.markdown("### Model Architecture")
        st.markdown(
            "| Model | EC | NLI | Notes |\n|-------|-----|-----|-------|\n"
            "| baseline | ✗ | ✗ | GNN → VerdictHead only |\n"
            "| v1-hgnn  | ✓ | ✗ | Symbolic override at threshold |\n"
            "| v2-hgnn  | ✓ | ✗ | EC scalar also feeds VerdictHead |\n"
            "| v3-nli   | ✓ | ✓ | DeBERTa-v3-small NLI replaces H1 |"
        )

        st.markdown("### Inference Strength (IS)")
        st.markdown(
            "IS is a regression head output ∈ [0, 1] representing how strongly "
            "the evidence logically supports or refutes the claim — independent of stance direction. "
            "High IS + strong stance → high EC → likely symbolic override."
        )

        st.markdown("### Verdict Classes")
        st.markdown(
            "| Label | Symbol | Meaning |\n|-------|--------|----------|\n"
            "| supported | ✓ | Evidence confirms the claim |\n"
            "| refuted | ✗ | Evidence contradicts the claim |\n"
            "| not_enough_evidence | ~ | Insufficient or conflicting evidence |"
        )


# ── Reports tab ───────────────────────────────────────────────────────────────

_REPORTS_ROOT = Path("out/reports/model")


def _load_verdict_metrics(model_key: str) -> dict | None:
    p = _REPORTS_ROOT / model_key / "eval" / "verdict_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_stance_metrics(model_key: str) -> dict | None:
    p = _REPORTS_ROOT / model_key / "eval" / "stance_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_is_metrics(model_key: str) -> dict | None:
    p = _REPORTS_ROOT / model_key / "eval" / "is_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_training_history(model_key: str) -> dict | None:
    p = _REPORTS_ROOT / model_key / "training_history.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _render_model_report(model_key: str) -> None:
    vm = _load_verdict_metrics(model_key)
    sm = _load_stance_metrics(model_key)
    im = _load_is_metrics(model_key)
    th = _load_training_history(model_key)

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
        # Support both old format (bare list) and new format (dict with "history" key)
        if isinstance(th, list):
            history, cov = th, None
        else:
            history = th.get("history", [])
            cov = th.get("data_coverage")

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

    # ── Verdict tabs: breakdown / plots ──────────────────────────────────────
    t_plots, t_verdict, t_stance, t_is, t_skipped = st.tabs(
        ["Plots", "Verdict", "Stance", "Inf. Strength", "Skipped IDs"]
    )

    with t_plots:
        plots_dir = _REPORTS_ROOT / model_key / "eval" / "plots"
        imgs = [
            (plots_dir / "confusion_matrix.png",   "Confusion Matrix"),
            (plots_dir / "class_f1.png",            "Per-Class F1"),
            (plots_dir / "per_source_accuracy.png", "Per-Source Accuracy"),
        ]
        cols = st.columns(len(imgs))
        for col, (img_path, caption) in zip(cols, imgs):
            if img_path.exists():
                col.image(str(img_path), caption=caption, use_container_width=True)
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
            st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

        conf = vm.get("confusion")
        if conf:
            st.markdown("##### Confusion Matrix")
            labels = list(vm.get("per_class", {}).keys()) or [str(i) for i in range(len(conf))]
            import pandas as pd
            st.dataframe(
                pd.DataFrame(conf, index=labels, columns=labels),
                use_container_width=True,
            )

        per_src = vm.get("per_source")
        if per_src:
            st.markdown("##### Per-Source Accuracy")
            import pandas as pd
            st.dataframe(
                pd.DataFrame(
                    [{"Source": s, "Accuracy": m["accuracy"], "N": m["support"]}
                     for s, m in per_src.items()]
                ).set_index("Source"),
                use_container_width=True,
            )

    with t_stance:
        if sm:
            ca, cb, cc = st.columns(3)
            ca.metric("Accuracy",  f"{sm['accuracy']:.1%}")
            cb.metric("Macro F1",  f"{sm['macro_f1']:.3f}")
            cc.metric("ECE",       f"{sm['ece']:.4f}")
            rows = [
                {"Class": cls, "Precision": m["precision"], "Recall": m["recall"],
                 "F1": m["f1"], "N": m["support"]}
                for cls, m in sm.get("per_class", {}).items()
            ]
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
        else:
            st.caption("stance_metrics.json not found.")

    with t_is:
        if im:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE",       f"{im['rmse']:.4f}")
            c2.metric("Pearson r",  f"{im['pearson_r']:.4f}")
            c3.metric("Pred Mean",  f"{im['pred_mean']:.4f}")
            c4.metric("True Mean",  f"{im['true_mean']:.4f}")
        else:
            st.caption("is_metrics.json not found.")

    with t_skipped:
        skipped_ids = vm.get("skipped_ids", [])
        if skipped_ids:
            st.caption(f"{len(skipped_ids)} claims skipped during evaluation.")
            import pandas as pd
            st.dataframe(
                pd.DataFrame({"Claim ID": skipped_ids}),
                use_container_width=True,
                height=min(400, 36 * len(skipped_ids) + 38),
            )
        else:
            st.success("No claims were skipped during evaluation.")


def _render_reports_tab() -> None:
    model_keys = list(_MODELS.keys())
    tabs = st.tabs([f"  {k}  " for k in model_keys])
    for tab, mk in zip(tabs, model_keys):
        with tab:
            _render_model_report(mk)


# ── Evaluate tab ──────────────────────────────────────────────────────────────

def _render_evaluate_tab(selected_key: str, *, state_key: str = "eval") -> None:
    records = _load_test_records()
    if not records:
        st.warning(f"Test data not found.\n- `{_DATA_JSONL}`\n- `{_TEST_IDX}`")
        return

    st.caption(f"{len(records)} test records available")
    models_to_eval = list(_MODELS.keys()) if selected_key == _ALL_KEY else [selected_key]

    c_n, c_seed = st.columns([3, 2])
    with c_n:
        n = st.slider("Samples", 5, min(len(records), 1000), 20, 5, key=f"{state_key}_samples")
    with c_seed:
        fixed = st.checkbox("Fixed seed", value=False, key=f"{state_key}_fixed_seed")
        seed  = st.number_input("Seed", value=42, step=1, label_visibility="collapsed", key=f"{state_key}_seed") if fixed else None

    if st.button("▶ Run", type="primary", key=f"{state_key}_run_btn"):
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
                                 "pred": None, "result": None, "source": source,
                                 "error": str(exc)})

        prog.empty()
        st.session_state[f"{state_key}_rows"]   = rows
        st.session_state[f"{state_key}_inspect"] = None

    rows = st.session_state.get(f"{state_key}_rows")
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
        for lbl in _VERDICT_LABELS:
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
            for i, lbl in enumerate(_VERDICT_LABELS):
                fc[i].metric(f"{_VERDICT_META[lbl][0]} F1", f"{f1s[i]:.3f}", help=lbl)

    # ── download button ───────────────────────────────────────────────────────
    valid_rows = [r for r in rows if r.get("pred") is not None]
    if valid_rows:
        json_data = _build_eval_export(rows)
        c_dl, c_pat = st.columns([2, 5])
        with c_dl:
            st.download_button(
                "⬇ Download JSON",
                json_data,
                file_name="eval_results.json",
                mime="application/json",
                use_container_width=True,
                key=f"{state_key}_dl_btn",
            )
        with c_pat:
            patterns = Counter(_failure_pattern(r) for r in rows)
            pat_parts = []
            for pat in ["correct", "ec_override_fix", "all_neutral_ec",
                        "placeholder_ev", "qa_format_ev", "genuine_error", "error"]:
                n = patterns.get(pat, 0)
                if n:
                    pat_parts.append(f"`{pat}` ×{n}")
            st.caption("  ·  ".join(pat_parts))

    st.divider()

    # ── inspection table ─────────────────────────────────────────────────────
    st.markdown("**Predictions** — expand to trace layer-wise reasoning")
    inspect_model = all_models[0] if len(all_models) == 1 else st.selectbox(
        "Show predictions for", all_models, label_visibility="collapsed",
        key=f"{state_key}_inspect_model",
    )
    mrows = [r for r in rows if r["model"] == inspect_model]

    for row in mrows[:50]:
        if row["pred"] is None:
            continue
        ok     = row["pred"] == row["true"]
        t_icon = _VERDICT_META.get(row["true"], ("?",))[0]
        p_icon = _VERDICT_META.get(row["pred"], ("?",))[0]
        marker = "✅" if ok else "❌"

        claim_snip = row["claim"][:70] + ("…" if len(row["claim"]) > 70 else "")
        with st.expander(
            f"{marker}  {row['source']}  ·  {claim_snip}  [{t_icon}→{p_icon}]",
            expanded=False,
        ):
            if row["result"] is not None:
                t_a, t_t, t_d = st.tabs(["Architecture Flow", "Layer Table", "Debug"])
                with t_a:
                    _render_arch_flow(row["result"], inspect_model)
                with t_t:
                    _render_layerwise(row["result"], inspect_model, true_label=row["true"])
                with t_d:
                    _render_debug_view(row["result"], row["claim"])
            else:
                st.error(row.get("error", "no result"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Epistemic FactKG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    st.markdown(
        '<div class="page-header">'
        '<span class="page-title">Epistemic Claim Verifier</span>'
        '<span class="page-badge">Pramana · Neuro-Symbolic</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Evidence Confidence · Source Trust · Inference Strength · Symbolic override @ 0.35")

    from tabs import verify as tab_verify_mod
    from tabs import evaluate as tab_evaluate_mod
    from tabs import reports as tab_reports_mod
    from tabs import data as tab_data_mod
    from tabs import registry as tab_registry_mod
    from tabs import reference as tab_reference_mod

    # ── 5-tab layout ────────────────────────────────────────────────────────────
    # Verify      — live claim verification (primary action)
    # Evaluate    — batch evaluation, single model OR all models (Compare mode)
    # Reports     — training history, per-model eval metrics, plots
    # Dataset     — data statistics, schema, claim browser
    # Knowledge   — source trust registry + formula reference + assumptions
    tab_v, tab_e, tab_r, tab_d, tab_k = st.tabs(
        ["🔍 Verify", "📊 Evaluate", "📈 Reports", "🗄 Dataset", "📚 Knowledge"]
    )

    with tab_v:
        tab_verify_mod.render()

    with tab_e:
        # Radio toggle: single model vs all-model compare
        mode = st.radio(
            "Mode", ["Single Model", "Compare All Models"],
            horizontal=True, label_visibility="collapsed", key="eval_mode_toggle"
        )
        st.markdown("---")
        if mode == "Single Model":
            tab_evaluate_mod.render_evaluate_tab()
        else:
            st.caption("Run all 4 models on the same random sample and compare side-by-side.")
            tab_evaluate_mod.render_compare_tab()

    with tab_r:
        st.markdown("### Model Reports")
        st.caption("Metrics, plots, and training history from the last pipeline run.")
        tab_reports_mod.render()

    with tab_d:
        tab_data_mod.render()

    with tab_k:
        # Sub-tabs: Registry view + Formula reference + Assumptions
        k_reg, k_ref = st.tabs(["🗂 Source Registry", "📖 Reference & Assumptions"])
        with k_reg:
            tab_registry_mod.render()
        with k_ref:
            tab_reference_mod.render()


if __name__ == "__main__":
    main()

