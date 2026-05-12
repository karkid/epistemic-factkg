"""Node/edge type enums and graph container for EpistemicHGNN (ADR-013, ADR-014)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch_geometric.data import HeteroData


class NodeType(StrEnum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    EPISTEMIC = "epistemic"
    TRIPLE = "triple"


class EdgeType(StrEnum):
    HAS_EVIDENCE = "has_evidence"
    SUPPORTS = "supports"
    REFUTES = "refutes"
    ABSENT = "absent"  # AI2THOR: simulation confirmed the state/object is absent
    NO_EVIDENCE = (
        "no_evidence"  # AVeriTeC NEE: web search found no answer (stance=null in JSON)
    )
    HAS_EPISTEMIC = "has_epistemic"
    HAS_TRIPLE = "has_triple"
    FROM_TRIPLE = "from_triple"


# Verdict label mapping (ADR-015: 3-class)
VERDICT_TO_INT: dict[str, int] = {
    "supported": 0,
    "refuted": 1,
    "not_enough_evidence": 2,
}

# Pramana one-hot index (5 training types, ADR-011)
PRAMANA_TO_INT: dict[str, int] = {
    "perception": 0,
    "testimony": 1,
    "non_apprehension": 2,
    "comparison_analogy": 3,
    "inference": 4,
}

# Modality one-hot index (ADR-014)
MODALITY_TO_INT: dict[str, int] = {
    "simulation_state": 0,
    "web_text": 1,
    "video": 2,
    "audio": 3,
    "image": 4,
}

NUM_PRAMANA = len(PRAMANA_TO_INT)  # 5
NUM_MODALITY = len(MODALITY_TO_INT)  # 5
NUM_VERDICT = len(VERDICT_TO_INT)  # 3


@dataclass
class ClaimGraph:
    """A single claim's heterogeneous subgraph with its integer verdict label."""

    data: HeteroData
    label: int
    pramana: str  # pramana_primary — for Phase 6 per-Pramana evaluation
    dataset: str  # provenance.dataset — for Phase 6 per-source evaluation

    def to(self, device: torch.device) -> ClaimGraph:
        return ClaimGraph(
            data=self.data.to(device),
            label=self.label,
            pramana=self.pramana,
            dataset=self.dataset,
        )
