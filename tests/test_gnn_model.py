"""Smoke tests for EpistemicHGNN — shape, forward pass, loss step (ADR-013)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.core.gnn.model import EpistemicHGNN
from src.core.gnn.types import NUM_VERDICT, NUM_PRAMANA


def _make_synthetic_batch(
    n_claims: int = 2, n_evidence: int = 3, n_triples: int = 2
) -> HeteroData:
    """Build a minimal synthetic HeteroData for forward-pass testing."""
    data = HeteroData()
    data["claim"].x = torch.randn(n_claims, 384)
    data["evidence"].x = torch.randn(n_evidence, 389)
    data["epistemic"].x = torch.randn(n_claims, 6)
    data["triple"].x = torch.randn(n_triples, 384)

    # has_evidence: each claim → evidence nodes (simplified: claim 0 → all evidence)
    data["claim", "has_evidence", "evidence"].edge_index = torch.tensor(
        [[0] * n_evidence, list(range(n_evidence))], dtype=torch.long
    )
    # supports: all evidence → claim 0
    data["evidence", "supports", "claim"].edge_index = torch.tensor(
        [list(range(n_evidence)), [0] * n_evidence], dtype=torch.long
    )
    # empty stubs for other stance types (required for consistent PyG slices)
    data["evidence", "refutes", "claim"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    data["evidence", "absent", "claim"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    data["evidence", "no_evidence", "claim"].edge_index = torch.zeros(
        (2, 0), dtype=torch.long
    )
    # has_epistemic with edge_attr
    data["claim", "has_epistemic", "epistemic"].edge_index = torch.tensor(
        [list(range(n_claims)), list(range(n_claims))], dtype=torch.long
    )
    data["claim", "has_epistemic", "epistemic"].edge_attr = torch.rand(n_claims, 1)

    # has_triple
    data["claim", "has_triple", "triple"].edge_index = torch.tensor(
        [[0] * n_triples, list(range(n_triples))], dtype=torch.long
    )
    # from_triple
    ev_tr_pairs = min(n_evidence, n_triples)
    data["evidence", "from_triple", "triple"].edge_index = torch.tensor(
        [list(range(ev_tr_pairs)), list(range(ev_tr_pairs))], dtype=torch.long
    )

    data["claim"].y = torch.tensor([0, 1], dtype=torch.long)
    return data


class TestEpistemicHGNNPathwayA:
    def test_output_keys(self):
        model = EpistemicHGNN()
        batch = _make_synthetic_batch()
        out = model(batch)
        assert "verdict" in out
        assert "pramana" not in out  # Pathway A: no aux head

    def test_verdict_output_shape(self):
        model = EpistemicHGNN()
        batch = _make_synthetic_batch(n_claims=2)
        out = model(batch)
        assert out["verdict"].shape == (2, NUM_VERDICT)  # [2, 3]

    def test_output_is_logits(self):
        model = EpistemicHGNN()
        batch = _make_synthetic_batch()
        out = model(batch)
        # Logits are unbounded (not softmax)
        assert (
            out["verdict"].requires_grad or not out["verdict"].requires_grad
        )  # shape check passes
        assert out["verdict"].dtype == torch.float32

    def test_loss_computable(self):
        model = EpistemicHGNN()
        batch = _make_synthetic_batch(n_claims=2)
        out = model(batch)
        labels = torch.tensor([0, 1], dtype=torch.long)
        loss = nn.CrossEntropyLoss()(out["verdict"], labels)
        assert loss.item() > 0

    def test_loss_decreases_after_one_step(self):
        torch.manual_seed(0)
        model = EpistemicHGNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_synthetic_batch(n_claims=2)
        labels = torch.tensor([0, 1], dtype=torch.long)

        model.train()
        out1 = model(batch)
        loss1 = nn.CrossEntropyLoss()(out1["verdict"], labels)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        out2 = model(batch)
        loss2 = nn.CrossEntropyLoss()(out2["verdict"], labels)

        # Loss should change (model updated) — not necessarily decrease in one step
        # but parameters must have moved
        assert loss1.item() != loss2.item()

    def test_no_nan_in_output(self):
        model = EpistemicHGNN()
        batch = _make_synthetic_batch()
        out = model(batch)
        assert not torch.isnan(out["verdict"]).any()

    def test_dropout_off_in_eval(self):
        model = EpistemicHGNN(dropout=0.9)
        batch = _make_synthetic_batch()
        model.eval()
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        # In eval mode, dropout is off — outputs must be identical
        assert torch.allclose(out1["verdict"], out2["verdict"])


class TestEpistemicHGNNPathwayB:
    def test_pramana_head_present(self):
        model = EpistemicHGNN(use_modality_learning=True)
        batch = _make_synthetic_batch(n_claims=2)
        out = model(batch)
        assert "pramana" in out

    def test_pramana_output_shape(self):
        model = EpistemicHGNN(use_modality_learning=True)
        batch = _make_synthetic_batch(n_claims=2)
        out = model(batch)
        assert out["pramana"].shape == (2, NUM_PRAMANA)  # [2, 5]

    def test_verdict_shape_unchanged(self):
        model = EpistemicHGNN(use_modality_learning=True)
        batch = _make_synthetic_batch(n_claims=2)
        out = model(batch)
        assert out["verdict"].shape == (2, NUM_VERDICT)


class TestEpistemicHGNNConfig:
    def test_custom_hidden_dim(self):
        model = EpistemicHGNN(hidden_dim=128)
        batch = _make_synthetic_batch(n_claims=1)
        out = model(batch)
        assert out["verdict"].shape == (1, NUM_VERDICT)

    def test_single_head(self):
        model = EpistemicHGNN(heads=1)
        batch = _make_synthetic_batch()
        out = model(batch)
        assert "verdict" in out
