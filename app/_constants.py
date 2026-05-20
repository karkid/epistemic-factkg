"""Shared constants — paths, model metadata, verdict/modality mappings."""
from __future__ import annotations

import json
from pathlib import Path

from src.epistemic.enums import Modality
from src.model.data.types import VERDICT_TO_INT, STANCE_TO_INT


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_JSONL      = Path("out/data/training/epistemic_factkg_training.jsonl")
TEST_IDX        = Path("out/data/splits/test_indices.json")
SPLITS_DIR      = Path("out/data/splits")
REPORTS_ROOT    = Path("out/reports/model")
GRAPH_CACHE_DIR = Path("out/model/graphs")
DATA_REPORT_DIR = Path("out/reports/data")
REGISTRY_PATH   = Path("data/registry/source_trust_registry.jsonl")
SCHEMA_PATH     = None  # schema lives in src/epistemic/schema.py:CLAIM_SCHEMA (no JSON file)
UNIFIED_JSONL   = Path("out/data/unified/epistemic_factkg.jsonl")


# ── Model labels (read live accuracy from verdict_metrics.json) ───────────────

_MODEL_ORDER = [
    ("v3-nli",   "NLI + Hybrid  "),
    ("v2-hgnn",  "Hybrid        "),
    ("v1-hgnn",  "Pure Symbolic "),
    ("baseline", "No EC         "),
]

ALL_KEY = "all"


def build_model_labels() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, desc in _MODEL_ORDER:
        try:
            acc = json.loads(
                (REPORTS_ROOT / key / "eval" / "verdict_metrics.json").read_text()
            )["accuracy"]
            acc_str = f"{acc:.1%}"
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            acc_str = "—"
        result[key] = f"{key} — {desc}·  {acc_str}"
    return result


MODELS: dict[str, str] = build_model_labels()

MODEL_DESCRIPTIONS: dict[str, str] = {
    "baseline": "GNN encoder → VerdictHead only. No epistemic confidence formula.",
    "v1-hgnn":  "Adds EC formula. H1 stance + H2 IS feed a symbolic decision at 0.35.",
    "v2-hgnn":  "Hybrid: EC aggregation jointly feeds VerdictHead as extra input.",
    "v3-nli":   "v2-hgnn + frozen DeBERTa-v3-small NLI replaces H1 in the EC path.",
}


# ── Verdict & stance metadata ─────────────────────────────────────────────────
# VERDICT_LABELS and INT_TO_VERDICT derive from VERDICT_TO_INT (model's 3-class output).
# Adding a verdict class to the model automatically updates display everywhere.

VERDICT_LABELS = list(VERDICT_TO_INT.keys())        # ["supported", "refuted", "not_enough_evidence"]
INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}   # {0: "supported", 1: "refuted", 2: "not_enough_evidence"}

# Model stance head predicts 3 classes; class 2 groups NEE + conflicting as "neutral" for display.
INT_TO_STANCE = {i: label for label, i in STANCE_TO_INT.items() if i not in {2}}
INT_TO_STANCE[2] = "neutral"

VERDICT_META: dict[str, tuple[str, str, str]] = {
    "supported":           ("✓", "SUPPORTED",           "#1d6340"),
    "refuted":             ("✗", "REFUTED",              "#9b2226"),
    "not_enough_evidence": ("~", "NOT ENOUGH EVIDENCE",  "#7f4f24"),
}
VERDICT_CSS     = {"supported": "sup", "refuted": "ref", "not_enough_evidence": "nei"}
STANCE_CHIP_CLS = {"supports": "chip-green", "refutes": "chip-red", "neutral": "chip-gray"}


# ── Modality / source metadata ────────────────────────────────────────────────

MODALITIES = [m.value for m in Modality]
MODALITY_LABELS = {
    Modality.WEB_TEXT:           "Web Text",
    Modality.PDF:                "PDF",
    Modality.IMAGE:              "Image",
    Modality.VIDEO:              "Video",
    Modality.AUDIO:              "Audio",
    Modality.WEB_TABLE:          "Table",
    Modality.SENSOR:             "Sensor",
    Modality.ANNOTATOR_KNOWLEDGE:"Annotator Knowledge",
    Modality.UNANSWERABLE:       "Unanswerable",
    Modality.OTHER:              "Other",
}
PRAMANA_SHORT = {
    Modality.WEB_TEXT:           "Shabda",
    Modality.PDF:                "Shabda",
    Modality.IMAGE:              "Pratyaksha",
    Modality.VIDEO:              "Pratyaksha",
    Modality.AUDIO:              "Pratyaksha",
    Modality.SENSOR:             "Pratyaksha",
    Modality.WEB_TABLE:          "Upamana",
    Modality.ANNOTATOR_KNOWLEDGE:"Shabda",
    Modality.UNANSWERABLE:       "—",
    Modality.OTHER:              "Shabda",
}
# The 6 one-hot encoder categories from SOURCE_TYPE_TO_INT in src/model/data/types.py.
# Keep in sync with that dict.  Do NOT expand to raw registry source_types here —
# the predictor maps source_id → encoder category at inference time.
SOURCE_TYPES = [
    "news_media",
    "academic",
    "government",
    "social_media",
    "sensor",
    "unknown",
]
SOURCE_LABELS = {
    "news_media":   "News Media",
    "academic":     "Academic / Knowledge Graph",
    "government":   "Government",
    "social_media": "Social Media",
    "sensor":       "Sensor / Perception",
    "unknown":      "Unknown",
    # Legacy aliases — keep so old session state / serialised records fall back gracefully
    "news":       "News Media",
    "simulation": "Simulation (AI2THOR)",
}

# Pattern-based source_id → source_type mapping (checked in order; first match wins).
SOURCE_ID_TYPE_MAP: list[tuple[str, str]] = [
    ("academic", "academic"),
    ("wikipedia", "academic"),
    ("scholar", "academic"),
    ("pubmed", "academic"),
    ("arxiv", "academic"),
    ("news", "news_media"),
    ("reuters", "news_media"),
    ("bbc", "news_media"),
    ("cnn", "news_media"),
    ("guardian", "news_media"),
    ("government", "government"),
    ("_gov", "government"),
    ("social", "social_media"),
    ("twitter", "social_media"),
    ("reddit", "social_media"),
    ("sensor", "sensor"),
    ("ai2thor", "sensor"),
    ("simulation", "sensor"),
]


def source_id_to_type(source_id: str) -> str:
    """Map a raw source_id string to a SOURCE_TYPES category."""
    sid = source_id.lower()
    for pattern, stype in SOURCE_ID_TYPE_MAP:
        if pattern in sid:
            return stype
    return "unknown"
