"""All @st.cache_data / @st.cache_resource data-loading functions."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from _constants import (
    DATA_JSONL, TEST_IDX, REPORTS_ROOT, DATA_REPORT_DIR,
    REGISTRY_PATH, UNIFIED_JSONL, SPLITS_DIR, GRAPH_CACHE_DIR,
)
from predictor import EpistemicPredictor


# ── Test records ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_test_records() -> list[dict]:
    if not DATA_JSONL.exists() or not TEST_IDX.exists():
        return []
    with open(TEST_IDX, encoding="utf-8") as f:
        indices = set(json.load(f)["indices"])
    records: list[dict] = []
    with open(DATA_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in indices:
                records.append(json.loads(line))
    return records


@st.cache_data(show_spinner=False)
def load_all_records_indexed() -> dict[str, dict]:
    """All JSONL records keyed by `id` (row index as fallback)."""
    if not DATA_JSONL.exists():
        return {}
    result: dict[str, dict] = {}
    with open(DATA_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                result[r.get("id", str(i))] = r
    return result


@st.cache_data(show_spinner=False)
def load_all_records_list() -> list[dict]:
    """All training JSONL records as a list (for search/filter)."""
    if not DATA_JSONL.exists():
        return []
    return [
        json.loads(l) for l in DATA_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


# ── Model predictor ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_predictor(model_name: str) -> EpistemicPredictor | str:
    try:
        return EpistemicPredictor(model_name)
    except FileNotFoundError as e:
        return str(e)


# ── Dataset stats (for Data tab) ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset_stats() -> dict | None:
    """Load from validation.json + training_validation.json."""
    val_path  = DATA_REPORT_DIR / "validation.json"
    train_path = DATA_REPORT_DIR / "training_validation.json"

    result: dict = {}

    if val_path.exists():
        try:
            val = json.loads(val_path.read_text(encoding="utf-8"))
            summaries = val.get("summaries", [])
            if summaries:
                s = summaries[0]
                result["counts"]        = s.get("counts", {})
                result["coverage"]      = s.get("coverage", {})
                result["distributions"] = s.get("distributions", {})
                result["schema_errors"] = s.get("schema_errors_top", {})
                result["logic_warnings"] = s.get("logic_warnings_top", {})
        except Exception:
            pass

    if train_path.exists():
        try:
            train = json.loads(train_path.read_text(encoding="utf-8"))
            result["training"] = train
        except Exception:
            pass

    # Load split sizes from index files
    splits: dict[str, int] = {}
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}_indices.json"
        if p.exists():
            try:
                splits[split] = len(json.loads(p.read_text(encoding="utf-8"))["indices"])
            except Exception:
                pass
    result["splits"] = splits

    return result if result else None


# ── Source trust registry ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return [
        json.loads(l) for l in REGISTRY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


# ── Graph cache (pkl) ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading graph cache…")
def load_graph_cache(model_key: str) -> dict | None:
    """Load split_cache_{model}.pkl.  Cached as resource (stays in memory)."""
    p = GRAPH_CACHE_DIR / f"split_cache_{model_key}.pkl"
    if not p.exists():
        return None
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def build_graph_id_map(model_key: str) -> dict[str, tuple[str, int]]:
    """Return {claim_id: (split, graph_index)} via sequential scan.

    The pkl stores graphs in the same order as split_indices, skipping
    records whose builder returned None.  We approximate the skip
    detection with a fast heuristic: records whose evidence list has no
    non-empty text would have produced None from the builder.
    """
    cache = load_graph_cache(model_key)
    if not cache:
        return {}

    if not DATA_JSONL.exists():
        return {}
    records = [
        json.loads(l)
        for l in DATA_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    result: dict[str, tuple[str, int]] = {}
    for split in ("train", "val"):
        idx_path = SPLITS_DIR / f"{split}_indices.json"
        if not idx_path.exists():
            continue
        split_indices = json.loads(idx_path.read_text(encoding="utf-8"))["indices"]
        graphs = cache.get(split, [])

        graph_pos = 0
        for row_idx in split_indices:
            if graph_pos >= len(graphs):
                break
            if row_idx >= len(records):
                continue
            rec = records[row_idx]
            has_ev = any(
                (e.get("text") or "").strip()
                for e in rec.get("evidence", [])
            )
            if not has_ev:
                continue  # builder would have returned None
            cid = rec.get("id", str(row_idx))
            result[cid] = (split, graph_pos)
            graph_pos += 1

    return result


# ── Per-model report files ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_verdict_metrics(model_key: str) -> dict | None:
    p = REPORTS_ROOT / model_key / "eval" / "verdict_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_stance_metrics(model_key: str) -> dict | None:
    p = REPORTS_ROOT / model_key / "eval" / "stance_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_is_metrics(model_key: str) -> dict | None:
    p = REPORTS_ROOT / model_key / "eval" / "is_metrics.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_training_history(model_key: str) -> dict | list | None:
    p = REPORTS_ROOT / model_key / "training_history.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
