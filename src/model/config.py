"""GraphConfig — single source of truth for node dims and edge types.

V1 graph schema:
  Nodes : claim (390-d), evidence (400-d), triple (384-d, AI2THOR only)
  Edges : has_evidence, connected_to, co_evidence, has_triple, from_triple

To extend to V2 (e.g. add source nodes):
    config.node_dims["source"] = 10
    config.edge_types.append(("evidence", "from_source", "source"))
Model code (EpistemicEncoder) reads these at construction — no other changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.model.data.types import (
    CLAIM_DIM,
    EVIDENCE_DIM,
    TRIPLE_DIM,
    EdgeType,
    NodeType,
)

EdgeTypeTuple = tuple[str, str, str]


@dataclass
class GraphConfig:
    node_dims: dict[str, int]
    edge_types: list[EdgeTypeTuple]
    target_node: str = NodeType.EVIDENCE
    symbolic_fields: list[str] = field(default_factory=lambda: ["ew", "st"])

    @classmethod
    def v1(cls) -> GraphConfig:
        return cls(
            node_dims={
                NodeType.CLAIM: CLAIM_DIM,  # 390
                NodeType.EVIDENCE: EVIDENCE_DIM,  # 400
                NodeType.TRIPLE: TRIPLE_DIM,  # 384
            },
            edge_types=[
                (NodeType.CLAIM, EdgeType.HAS_EVIDENCE, NodeType.EVIDENCE),
                (NodeType.EVIDENCE, EdgeType.CONNECTED_TO, NodeType.CLAIM),
                (NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE),
                (NodeType.CLAIM, EdgeType.HAS_TRIPLE, NodeType.TRIPLE),
                (NodeType.EVIDENCE, EdgeType.FROM_TRIPLE, NodeType.TRIPLE),
            ],
        )
