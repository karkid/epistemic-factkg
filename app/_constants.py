"""Shared constants — paths, model metadata, verdict/modality mappings."""
from __future__ import annotations

import json
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_JSONL      = Path("out/data/training/epistemic_factkg_training.jsonl")
TEST_IDX        = Path("out/data/splits/test_indices.json")
SPLITS_DIR      = Path("out/data/splits")
REPORTS_ROOT    = Path("out/reports/model")
GRAPH_CACHE_DIR = Path("out/model/graphs")
DATA_REPORT_DIR = Path("out/reports/data")
REGISTRY_PATH   = Path("data/registry/source_trust_registry.jsonl")
SCHEMA_PATH     = Path("data/schema/unified_schema.json")
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

VERDICT_META: dict[str, tuple[str, str, str]] = {
    "supported":           ("✓", "SUPPORTED",           "#1d6340"),
    "refuted":             ("✗", "REFUTED",              "#9b2226"),
    "not_enough_evidence": ("~", "NOT ENOUGH EVIDENCE",  "#7f4f24"),
}
VERDICT_LABELS  = ["supported", "refuted", "not_enough_evidence"]
VERDICT_CSS     = {"supported": "sup", "refuted": "ref", "not_enough_evidence": "nei"}
STANCE_CHIP_CLS = {"supports": "chip-green", "refutes": "chip-red", "neutral": "chip-gray"}


# ── Modality / source metadata ────────────────────────────────────────────────

MODALITIES = ["web_text", "pdf", "image", "video", "audio", "web_table"]
MODALITY_LABELS = {
    "web_text":  "Web Text",
    "pdf":       "PDF",
    "image":     "Image",
    "video":     "Video",
    "audio":     "Audio",
    "web_table": "Table",
}
PRAMANA_SHORT = {
    "web_text":  "Shabda",
    "pdf":       "Shabda",
    "image":     "Pratyaksha",
    "video":     "Pratyaksha",
    "audio":     "Pratyaksha",
    "web_table": "Upamana",
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
