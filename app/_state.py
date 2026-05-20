"""Session state helpers, record loading, model selector widget."""
from __future__ import annotations

import random

import streamlit as st

from _constants import MODELS, ALL_KEY, MODEL_DESCRIPTIONS, MODALITIES, SOURCE_TYPES, source_id_to_type
from _loaders import load_test_records, load_all_records_indexed, source_type_index


# ── Blank evidence item ───────────────────────────────────────────────────────

def blank_ev() -> dict:
    return {"text": "", "source_type": "unknown", "modality": "web_text"}


# ── Session state init ────────────────────────────────────────────────────────

def init_state() -> None:
    defaults: dict = {
        "evidence_list":      [blank_ev()],
        "last_claim":         "",
        "_random_true_label": None,
        "current_claim_id":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    pending = st.session_state.pop("_pending_claim", None)
    if pending is not None:
        st.session_state["claim_input"] = pending


# ── Record loading ────────────────────────────────────────────────────────────

def load_record_into_state(rec: dict) -> None:
    """Push a record's claim + evidence into session state for the Verify tab."""
    old_n = len(st.session_state.get("evidence_list", []))
    for i in range(old_n + 6):
        for pfx in ("ev_", "mod_", "src_"):
            st.session_state.pop(f"{pfx}{i}", None)
    new_evs = [
        {
            "text":        ev.get("text", ""),
            "source_id":   ev.get("source_id", ""),
            "source_type": source_type_index().get(ev.get("source_id", ""), source_id_to_type(ev.get("source_id", ""))),
            "modality":    ev.get("modality", "web_text"),
        }
        for ev in rec.get("evidence", [])[:4]
    ] or [blank_ev()]
    st.session_state["_pending_claim"]     = rec["claim"]
    st.session_state["last_claim"]         = rec["claim"]
    st.session_state["evidence_list"]      = new_evs
    st.session_state["_random_true_label"] = rec.get("verdict", {}).get("label")
    st.session_state["current_claim_id"]   = rec.get("id")
    for i, ev in enumerate(new_evs):
        st.session_state[f"ev_{i}"]  = ev["text"]
        st.session_state[f"mod_{i}"] = ev["modality"]
        st.session_state[f"src_{i}"] = ev["source_type"]


def load_random_example() -> None:
    records = load_test_records()
    if not records:
        st.warning("Test data not found.")
        return
    load_record_into_state(random.choice(records))


def load_by_id(claim_id: str) -> bool:
    """Load record by ID. Returns True on success."""
    idx_map = load_all_records_indexed()
    rec = idx_map.get(claim_id.strip())
    if rec is None:
        return False
    load_record_into_state(rec)
    return True


# ── Model selector widget ─────────────────────────────────────────────────────

def model_selector(widget_key: str, allow_all: bool = False) -> str:
    """Compact horizontal model radio; returns the selected model key."""
    keys   = list(MODELS.keys())
    labels = list(MODELS.values())
    if allow_all:
        keys.append(ALL_KEY)
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
    desc = MODEL_DESCRIPTIONS.get(selected, "")
    if desc:
        st.caption(desc)
    return selected
