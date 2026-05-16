"""Node/edge type constants and graph container for EpistemicHGNN (V1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch_geometric.data import HeteroData

_EMBED_DIM = 384

class NodeType(StrEnum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    TRIPLE = "triple"


class EdgeType(StrEnum):
    HAS_EVIDENCE = "has_evidence"  # claim → evidence
    CONNECTED_TO = "connected_to"  # evidence → claim (neutral reverse)
    CO_EVIDENCE = "co_evidence"  # evidence → evidence (fully connected within claim)
    HAS_TRIPLE = "has_triple"  # claim → triple (AI2THOR only)
    FROM_TRIPLE = "from_triple"  # evidence → triple (AI2THOR only)


# ── Verdict ──────────────────────────────────────────────────────────────────

VERDICT_TO_INT: dict[str, int] = {
    "supported": 0,
    "refuted": 1,
    "not_enough_evidence": 2,
}
NUM_VERDICT = len(VERDICT_TO_INT)  # 3


# ── Stance (H1 supervision labels) ───────────────────────────────────────────
# absent maps to supports: sensor-confirmed absence is positive evidence for
# claims about missing objects (contributes to support score in EC aggregation).
# not_enough_evidence and conflicting_evidence map to neutral.

STANCE_TO_INT: dict[str, int] = {
    "supports": 0,
    "absent": 0,
    "refutes": 1,
    "not_enough_evidence": 2,
    "conflicting_evidence": 2,
}
NUM_STANCE = 3  # supports / refutes / neutral


# ── Evidence type (multi-hot, 5 training types) ───────────────────────────────

EVIDENCE_TYPE_TO_INT: dict[str, int] = {
    "perception": 0,
    "testimony": 1,
    "non_apprehension": 2,
    "comparison_analogy": 3,
    "inference": 4,
}
NUM_EVIDENCE_TYPE = len(EVIDENCE_TYPE_TO_INT)  # 5


# ── Reasoning strategy (one-hot, claim node) ─────────────────────────────────

REASONING_STRATEGY_TO_INT: dict[str, int] = {
    "direct_observation": 0,
    "absence_detection": 1,
    "spatial_comparison": 2,
    "testimonial_lookup": 3,
    "multi_hop_inference": 4,
    "conflicting_evidence": 5,
}
NUM_REASONING_STRATEGY = len(REASONING_STRATEGY_TO_INT)  # 6


# ── Modality (one-hot, evidence node) ────────────────────────────────────────

MODALITY_TO_INT: dict[str, int] = {
    "simulation_state": 0,
    "web_text": 1,
    "video": 2,
    "audio": 3,
    "image": 4,
}
NUM_MODALITY = len(MODALITY_TO_INT)  # 5


# ── Source type (one-hot, evidence node) ─────────────────────────────────────

SOURCE_TYPE_TO_INT: dict[str, int] = {
    "news_media": 0,
    "academic": 1,
    "government": 2,
    "social_media": 3,
    "simulation": 4,
    "unknown": 5,
}
NUM_SOURCE_TYPE = len(SOURCE_TYPE_TO_INT)  # 6

# Maps registry source_type values → the 6 encoder categories
_REGISTRY_TYPE_TO_CATEGORY: dict[str, str] = {
    "news_media": "news_media",
    "fact_checker": "news_media",
    "testimony": "news_media",
    "government": "government",
    "scientific_paper": "academic",
    "knowledge_graph": "academic",
    "social_media": "social_media",
    "simulation": "simulation",
}


def get_source_category(source_id: str, registry: dict[str, dict]) -> str:
    """Resolve a source_id to one of the 6 encoder source-type categories."""
    entry = registry.get(source_id, {})
    source_type = entry.get("source_type", "unknown")
    return _REGISTRY_TYPE_TO_CATEGORY.get(source_type, "unknown")


# ── Node feature dimensions ───────────────────────────────────────────────────

CLAIM_DIM = _EMBED_DIM + NUM_REASONING_STRATEGY  # 390
EVIDENCE_DIM = _EMBED_DIM + NUM_MODALITY + NUM_EVIDENCE_TYPE + NUM_SOURCE_TYPE  # 400
TRIPLE_DIM = _EMBED_DIM


# ── Graph container ───────────────────────────────────────────────────────────


@dataclass
class ClaimGraph:
    """A single claim's heterogeneous subgraph.

    data["evidence"].stance_y  — int tensor [N_ev]:   H1 supervision labels
    data["evidence"].is_y      — float tensor [N_ev]: H2 supervision targets
    data["evidence"].ew        — float tensor [N_ev]: epistemic weight (not encoder input)
    data["evidence"].st        — float tensor [N_ev]: source trust (not encoder input)
    """

    data: HeteroData
    label: int  # verdict integer for final eval
    dataset: str  # provenance.dataset for per-source breakdown

    def to(self, device: torch.device) -> ClaimGraph:
        return ClaimGraph(
            data=self.data.to(device),
            label=self.label,
            dataset=self.dataset,
        )
