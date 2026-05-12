"""Converts a unified JSONL record to a PyG HeteroData subgraph (ADR-013, ADR-014)."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.core.gnn.featurizer import Featurizer
from src.core.gnn.types import (
    ClaimGraph,
    PRAMANA_TO_INT,
    VERDICT_TO_INT,
)


class ClaimGraphBuilder:
    """Builds one ClaimGraph per unified record.

    Node features (ADR-014):
      claim:     sentence embedding (384-d)
      evidence:  sentence embedding (384-d) + modality one-hot (5-d) = 389-d
      epistemic: pramana one-hot (5-d) + confidence_weight (1-d) = 6-d
      triple:    sentence embedding of "s p o" string (384-d)  [AI2THOR only]

    Edge types (ADR-014):
      (claim, has_evidence, evidence)   — all records
      (evidence, supports, claim)       — stance == supports
      (evidence, refutes, claim)        — stance == refutes
      (evidence, absent, claim)         — stance == absent
      (claim, has_epistemic, epistemic) — all records; edge_attr = confidence_weight
      (claim, has_triple, triple)       — AI2THOR only
      (evidence, from_triple, triple)   — AI2THOR only
    """

    def __init__(self, featurizer: Featurizer):
        self._feat = featurizer

    def build(self, record: dict) -> ClaimGraph:
        data = HeteroData()

        # ── Claim node ────────────────────────────────────────────────────────
        claim_text = record.get("claim", "")
        data["claim"].x = self._feat.encode_texts([claim_text])  # [1, 384]

        # pramana_y: integer Pramana label for Pathway B auxiliary head (ADR-016)
        pramana_primary_for_label = record.get("epistemic", {}).get("pramana_primary", "")
        data["claim"].pramana_y = torch.tensor(
            [PRAMANA_TO_INT.get(pramana_primary_for_label, -1)], dtype=torch.long
        )

        # ── Evidence nodes ────────────────────────────────────────────────────
        evidence_items = record.get("evidence") or []
        ev_texts = [e.get("text") or "" for e in evidence_items]
        ev_modalities = [e.get("modality") or "" for e in evidence_items]
        ev_stances = [
            e.get("stance") for e in evidence_items
        ]  # preserve None (→ no_evidence edge)

        if ev_texts:
            ev_embeddings = self._feat.encode_texts(ev_texts)  # [N, 384]
            modality_vecs = torch.stack(
                [self._feat.encode_modality(m) for m in ev_modalities]  # [N, 5]
            )
            data["evidence"].x = torch.cat(
                [ev_embeddings, modality_vecs], dim=1
            )  # [N, 389]
        else:
            data["evidence"].x = torch.zeros((0, 389), dtype=torch.float32)

        n_ev = len(ev_texts)

        # ── Epistemic node ─────────────────────────────────────────────────────
        pramana_primary = record.get("epistemic", {}).get("pramana_primary", "")
        confidence_weight = float(
            record.get("epistemic", {}).get("confidence_weight", 0.0)
        )
        pramana_vec = self._feat.encode_pramana(pramana_primary)  # [5]
        conf_vec = torch.tensor([confidence_weight], dtype=torch.float32)  # [1]
        data["epistemic"].x = torch.cat([pramana_vec, conf_vec]).unsqueeze(0)  # [1, 6]

        # ── Triple nodes (AI2THOR only) ────────────────────────────────────────
        claim_triples = record.get("claim_triples") or []
        # claim_triples items are [s, p, o] lists (unified schema v2.0)
        triple_texts = [f"{t[0]} {t[1]} {t[2]}" for t in claim_triples]
        if triple_texts:
            data["triple"].x = self._feat.encode_texts(triple_texts)  # [T, 384]
        else:
            data["triple"].x = torch.zeros((0, 384), dtype=torch.float32)
        n_tr = len(triple_texts)

        # ── Edges ──────────────────────────────────────────────────────────────
        # (claim, has_evidence, evidence): claim 0 → each evidence node
        if n_ev > 0:
            data["claim", "has_evidence", "evidence"].edge_index = torch.tensor(
                [[0] * n_ev, list(range(n_ev))], dtype=torch.long
            )

            # Stance-typed reverse edges (evidence → claim).
            # null/None stance (JSON "stance": null) means the web search found no answer
            # — distinct from "absent" which is AI2THOR confirming a state is physically absent.
            # Normalised to "no_evidence" so every evidence item has a positive back edge.
            supports_idx = [i for i, s in enumerate(ev_stances) if s == "supports"]
            refutes_idx = [i for i, s in enumerate(ev_stances) if s == "refutes"]
            absent_idx = [i for i, s in enumerate(ev_stances) if s == "absent"]
            no_evidence_idx = [i for i, s in enumerate(ev_stances) if s is None]

            for stance, idxs in [
                ("supports", supports_idx),
                ("refutes", refutes_idx),
                ("absent", absent_idx),
                ("no_evidence", no_evidence_idx),
            ]:
                if idxs:
                    data["evidence", stance, "claim"].edge_index = torch.tensor(
                        [idxs, [0] * len(idxs)], dtype=torch.long
                    )

        # (claim, has_epistemic, epistemic): edge_attr = confidence_weight (ADR-014)
        data["claim", "has_epistemic", "epistemic"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
        data["claim", "has_epistemic", "epistemic"].edge_attr = torch.tensor(
            [[confidence_weight]], dtype=torch.float32
        )

        # (claim, has_triple, triple) and (evidence, from_triple, triple): AI2THOR only
        if n_tr > 0:
            data["claim", "has_triple", "triple"].edge_index = torch.tensor(
                [[0] * n_tr, list(range(n_tr))], dtype=torch.long
            )
            # Link each triple back to the evidence node at the same index (if available)
            ev_to_triple = [(i, i) for i in range(min(n_ev, n_tr))]
            if ev_to_triple:
                src, dst = zip(*ev_to_triple)
                data["evidence", "from_triple", "triple"].edge_index = torch.tensor(
                    [list(src), list(dst)], dtype=torch.long
                )

        # Ensure all edge types are present in every graph so PyG InMemoryDataset
        # produces consistent slices across the collated batch. Graphs missing an
        # edge type (e.g. AVeriTeC records with no triples) would otherwise cause
        # slices shorter than N+1, breaking __getitem__ and DataLoader iteration.
        _required_edge_types = [
            ("claim", "has_evidence", "evidence"),
            ("evidence", "supports", "claim"),
            ("evidence", "refutes", "claim"),
            ("evidence", "absent", "claim"),
            ("evidence", "no_evidence", "claim"),
            ("claim", "has_epistemic", "epistemic"),
            ("claim", "has_triple", "triple"),
            ("evidence", "from_triple", "triple"),
        ]
        for _src, _rel, _dst in _required_edge_types:
            if (_src, _rel, _dst) not in data.edge_types:
                data[_src, _rel, _dst].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )

        # ── Label and metadata ─────────────────────────────────────────────────
        verdict_label = record.get("verdict", {}).get("label", "")
        label = VERDICT_TO_INT.get(verdict_label, -1)
        dataset = record.get("provenance", {}).get("dataset", "unknown")

        return ClaimGraph(
            data=data,
            label=label,
            pramana=pramana_primary,
            dataset=dataset,
        )
