"""Build per-claim HeteroData graphs for EpistemicHGNN (V1).

Graph structure per claim:
  Nodes:  claim (1), evidence (N_ev), triple (N_tr, AI2THOR only)
  Edges:  has_evidence, connected_to, co_evidence, has_triple, from_triple

No stance-typed edges — stance is learned by H1 from supervised labels.
EW and ST are stored as separate tensors (not encoder input features).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from src.core.claims.labels import (
    combine_evidence_weights,
    get_source_trust,
    load_source_trust_registry,
)
from src.core.gnn.featurizer import Featurizer
from src.core.gnn.types import (
    STANCE_TO_INT,
    VERDICT_TO_INT,
    ClaimGraph,
    EdgeType,
    NodeType,
    get_source_category,
)


class ClaimGraphBuilder:
    """Converts a v3.0 unified record dict into a PyG HeteroData subgraph."""

    def __init__(
        self,
        registry: dict[str, dict],
        featurizer: Featurizer,
    ) -> None:
        self.registry = registry
        self.featurizer = featurizer

    @classmethod
    def from_paths(
        cls,
        registry_path: str | Path,
        embed_cache_path: str | Path | None = None,
    ) -> ClaimGraphBuilder:
        registry = load_source_trust_registry(registry_path)
        featurizer = Featurizer(cache_path=embed_cache_path)
        return cls(registry, featurizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, record: dict) -> ClaimGraph:
        """Convert a single v3.0 record to a ClaimGraph.

        Returns None implicitly only if verdict is unknown — callers should
        filter to training evidence types before calling build().
        """
        data = HeteroData()
        evidence_items = record.get("evidence", [])

        # ── Claim node ────────────────────────────────────────────────
        strategy = self._extract_strategy(record)
        claim_text_emb = self.featurizer.encode_texts([record["claim"]])  # [1, 384]
        claim_strategy  = self.featurizer.encode_reasoning_strategy(strategy).unsqueeze(0)  # [1, 6]
        data[NodeType.CLAIM].x = torch.cat([claim_text_emb, claim_strategy], dim=1)  # [1, 390]

        # ── Evidence nodes ────────────────────────────────────────────
        n_ev = len(evidence_items)
        if n_ev == 0:
            # Degenerate graph — create a dummy evidence node so edges are valid
            data[NodeType.EVIDENCE].x = torch.zeros(1, 400, dtype=torch.float32)
            data[NodeType.EVIDENCE].stance_y = torch.tensor([2], dtype=torch.long)
            data[NodeType.EVIDENCE].is_y     = torch.tensor([0.0], dtype=torch.float32)
            data[NodeType.EVIDENCE].ew       = torch.tensor([0.0], dtype=torch.float32)
            data[NodeType.EVIDENCE].st       = torch.tensor([0.0], dtype=torch.float32)
            n_ev = 1
        else:
            ev_features, stance_y, is_y, ew_vals, st_vals = self._build_evidence(evidence_items)
            data[NodeType.EVIDENCE].x        = ev_features          # [N_ev, 400]
            data[NodeType.EVIDENCE].stance_y = stance_y             # [N_ev]
            data[NodeType.EVIDENCE].is_y     = is_y                 # [N_ev]
            data[NodeType.EVIDENCE].ew       = ew_vals              # [N_ev]
            data[NodeType.EVIDENCE].st       = st_vals              # [N_ev]

        # ── Triple nodes (AI2THOR only) ───────────────────────────────
        triple_texts, has_triple_idx, from_triple_idx = self._collect_triples(
            record, evidence_items
        )
        n_tr = len(triple_texts)
        if n_tr > 0:
            data[NodeType.TRIPLE].x = self.featurizer.encode_texts(triple_texts)  # [N_tr, 384]

        # ── Edges ─────────────────────────────────────────────────────
        ev_range = torch.arange(n_ev, dtype=torch.long)
        claim_idx = torch.zeros(n_ev, dtype=torch.long)

        # claim → evidence
        data[NodeType.CLAIM, EdgeType.HAS_EVIDENCE, NodeType.EVIDENCE].edge_index = \
            torch.stack([claim_idx, ev_range])  # [2, N_ev]

        # evidence → claim (neutral reverse)
        data[NodeType.EVIDENCE, EdgeType.CONNECTED_TO, NodeType.CLAIM].edge_index = \
            torch.stack([ev_range, claim_idx])  # [2, N_ev]

        # evidence → evidence (fully connected within claim)
        if n_ev > 1:
            src, dst = zip(*[(i, j) for i in range(n_ev) for j in range(n_ev) if i != j])
            data[NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE].edge_index = \
                torch.tensor([list(src), list(dst)], dtype=torch.long)
        else:
            data[NodeType.EVIDENCE, EdgeType.CO_EVIDENCE, NodeType.EVIDENCE].edge_index = \
                torch.zeros(2, 0, dtype=torch.long)

        # AI2THOR triple edges
        if n_tr > 0 and has_triple_idx:
            c_src, t_dst = zip(*has_triple_idx)
            data[NodeType.CLAIM, EdgeType.HAS_TRIPLE, NodeType.TRIPLE].edge_index = \
                torch.tensor([list(c_src), list(t_dst)], dtype=torch.long)
        if n_tr > 0 and from_triple_idx:
            e_src, t_dst = zip(*from_triple_idx)
            data[NodeType.EVIDENCE, EdgeType.FROM_TRIPLE, NodeType.TRIPLE].edge_index = \
                torch.tensor([list(e_src), list(t_dst)], dtype=torch.long)

        label = VERDICT_TO_INT.get(
            record.get("verdict", {}).get("label", "not_enough_evidence"), 2
        )
        dataset = record.get("provenance", {}).get("dataset", "unknown")
        return ClaimGraph(data=data, label=label, dataset=dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_strategy(self, record: dict) -> str:
        return (record.get("reasoning") or {}).get("strategy") or "testimonial_lookup"

    def _build_evidence(
        self, evidence_items: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        texts      = [ev.get("text", "") for ev in evidence_items]
        text_embs  = self.featurizer.encode_texts(texts)  # [N_ev, 384]

        modality_vecs, et_vecs, src_vecs = [], [], []
        stance_ints, is_vals, ew_vals, st_vals = [], [], [], []

        for ev in evidence_items:
            modality_vecs.append(self.featurizer.encode_modality(ev.get("modality")))
            et_vecs.append(self.featurizer.encode_evidence_types(ev.get("evidence_types", [])))
            cat = get_source_category(ev.get("source_id", "unknown_web"), self.registry)
            src_vecs.append(self.featurizer.encode_source_type(cat))

            stance_ints.append(STANCE_TO_INT.get(ev.get("stance", "not_enough_evidence"), 2))
            is_vals.append(float(ev.get("inference_strength", 0.6)))
            ew_vals.append(combine_evidence_weights(ev.get("evidence_types", [])))
            st_vals.append(get_source_trust(ev.get("source_id", "unknown_web"), self.registry))

        ev_features = torch.cat(
            [
                text_embs,
                torch.stack(modality_vecs),
                torch.stack(et_vecs),
                torch.stack(src_vecs),
            ],
            dim=1,
        )  # [N_ev, 400]

        return (
            ev_features,
            torch.tensor(stance_ints, dtype=torch.long),
            torch.tensor(is_vals,     dtype=torch.float32),
            torch.tensor(ew_vals,     dtype=torch.float32),
            torch.tensor(st_vals,     dtype=torch.float32),
        )

    def _collect_triples(
        self,
        record: dict,
        evidence_items: list[dict],
    ) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]]]:
        """Collect unique triple texts and build edge index lists.

        Returns:
            triple_texts:    List of "s p o" strings (deduplicated).
            has_triple_idx:  [(claim_idx=0, triple_idx), ...] for claim-level triples.
            from_triple_idx: [(ev_idx, triple_idx), ...] for evidence-level triples.
        """
        triple_to_idx: dict[str, int] = {}
        has_triple_idx: list[tuple[int, int]] = []
        from_triple_idx: list[tuple[int, int]] = []

        def _register(triple: list) -> int:
            key = f"{triple[0]} {triple[1]} {triple[2]}"
            if key not in triple_to_idx:
                triple_to_idx[key] = len(triple_to_idx)
            return triple_to_idx[key]

        # Claim-level triples → has_triple edges
        for triple in record.get("claim_triples") or []:
            if len(triple) == 3:
                t_idx = _register(triple)
                has_triple_idx.append((0, t_idx))

        # Evidence-level triples → from_triple edges
        for ev_idx, ev in enumerate(evidence_items):
            for triple in ev.get("triples") or []:
                if len(triple) == 3:
                    t_idx = _register(triple)
                    from_triple_idx.append((ev_idx, t_idx))

        return list(triple_to_idx.keys()), has_triple_idx, from_triple_idx
