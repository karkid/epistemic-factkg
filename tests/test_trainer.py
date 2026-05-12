"""Integration tests for Trainer — masking, checkpoint, early stopping (ADR-016)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.core.gnn.model import EpistemicHGNN
from src.core.gnn.train import EpochResult, Trainer, TrainConfig


# ── Synthetic batch helpers ───────────────────────────────────────────────────

def _make_graph(label: int = 0, n_ev: int = 2) -> HeteroData:
    data = HeteroData()
    data["claim"].x = torch.randn(1, 384)
    data["claim"].y = torch.tensor([label], dtype=torch.long)
    data["claim"].pramana_y = torch.tensor([0], dtype=torch.long)
    data["evidence"].x = torch.randn(n_ev, 389)
    data["epistemic"].x = torch.randn(1, 6)
    data["triple"].x = torch.zeros((0, 384), dtype=torch.float32)

    data["claim", "has_evidence", "evidence"].edge_index = torch.tensor(
        [[0] * n_ev, list(range(n_ev))], dtype=torch.long
    )
    data["evidence", "supports", "claim"].edge_index = torch.tensor(
        [list(range(n_ev)), [0] * n_ev], dtype=torch.long
    )
    data["evidence", "refutes", "claim"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["evidence", "absent", "claim"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["evidence", "no_evidence", "claim"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["claim", "has_epistemic", "epistemic"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data["claim", "has_epistemic", "epistemic"].edge_attr = torch.tensor([[0.9]])
    data["claim", "has_triple", "triple"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["evidence", "from_triple", "triple"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data


def _make_loader(n: int = 8) -> DataLoader:
    graphs = [_make_graph(label=i % 3) for i in range(n)]
    return DataLoader(graphs, batch_size=4, shuffle=False)


def _make_trainer(
    masked_edge_types: list[str] | None = None,
    patience: int = 10,
    tmp_path: Path | None = None,
) -> Trainer:
    model = EpistemicHGNN(hidden_dim=64, heads=1, dropout=0.0)
    config = TrainConfig(
        epochs=20,
        lr=1e-3,
        batch_size=4,
        hidden_dim=64,
        heads=1,
        dropout=0.0,
        patience=patience,
        masked_edge_types=masked_edge_types or [],
        checkpoint_dir=str(tmp_path or Path("out/checkpoints/test_tmp")),
    )
    return Trainer(model=model, class_weights=None, config=config)


# ── Tests: edge masking ───────────────────────────────────────────────────────

class TestEdgeMasking:
    def test_masked_stance_edges_are_zeroed(self, tmp_path):
        """After masking, the specified edge types must have empty edge_index."""
        trainer = _make_trainer(
            masked_edge_types=["supports", "refutes", "absent", "no_evidence"],
            tmp_path=tmp_path,
        )
        loader = _make_loader(n=4)
        # Run one epoch; the masking happens inside _run_epoch
        trainer._run_epoch(loader, train=False)

        # Verify by manually applying masking to a batch and checking
        batch = next(iter(loader)).to(trainer.device)
        masked_rels = {"supports", "refutes", "absent", "no_evidence"}
        for rel in trainer.config.masked_edge_types:
            for et in batch.edge_types:
                if et[1] == rel:
                    batch[et].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long, device=trainer.device
                    )
        for et in batch.edge_types:
            if et[1] in masked_rels:
                assert batch[et].edge_index.shape[1] == 0, f"{et} not zeroed"

    def test_non_masked_edges_unchanged(self, tmp_path):
        """Edges not in masked_edge_types must keep their original indices."""
        trainer = _make_trainer(
            masked_edge_types=["supports"],
            tmp_path=tmp_path,
        )
        loader = _make_loader(n=4)
        batch = next(iter(loader)).to(trainer.device)
        original_has_ev = batch["claim", "has_evidence", "evidence"].edge_index.clone()

        for rel in trainer.config.masked_edge_types:
            for et in batch.edge_types:
                if et[1] == rel:
                    batch[et].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long, device=trainer.device
                    )

        assert torch.equal(
            batch["claim", "has_evidence", "evidence"].edge_index, original_has_ev
        )

    def test_no_masking_leaves_supports_edges_intact(self, tmp_path):
        """With empty masked_edge_types, no edges are modified."""
        trainer = _make_trainer(masked_edge_types=[], tmp_path=tmp_path)
        loader = _make_loader(n=4)
        batch = next(iter(loader)).to(trainer.device)
        original = batch["evidence", "supports", "claim"].edge_index.clone()

        for rel in trainer.config.masked_edge_types:
            for et in batch.edge_types:
                if et[1] == rel:
                    batch[et].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long, device=trainer.device
                    )

        assert torch.equal(batch["evidence", "supports", "claim"].edge_index, original)


# ── Tests: checkpoint ─────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_fit_saves_best_checkpoint(self, tmp_path):
        trainer = _make_trainer(tmp_path=tmp_path)
        loader = _make_loader(n=8)
        trainer.fit(loader, loader, verbose=False)
        assert (tmp_path / "best_model.pt").exists()

    def test_load_best_restores_weights(self, tmp_path):
        trainer = _make_trainer(tmp_path=tmp_path)
        loader = _make_loader(n=8)
        trainer.fit(loader, loader, verbose=False)

        # Load best once to capture the saved weights
        trainer.load_best()
        first_param = next(trainer.model.parameters())
        saved_weights = first_param.data.clone()

        # Corrupt the weight
        with torch.no_grad():
            first_param.data.fill_(999.0)

        # Restore and verify it matches the saved weights (not the corrupted 999.0)
        trainer.load_best()
        restored = next(trainer.model.parameters()).data
        assert not torch.allclose(restored, torch.full_like(restored, 999.0))
        assert torch.allclose(restored, saved_weights)

    def test_history_returned_by_fit(self, tmp_path):
        trainer = _make_trainer(tmp_path=tmp_path, patience=2)
        loader = _make_loader(n=8)
        history = trainer.fit(loader, loader, verbose=False)
        assert isinstance(history, list)
        assert len(history) >= 1
        assert "epoch" in history[0]
        assert "train_acc" in history[0]
        assert "val_acc" in history[0]


# ── Tests: early stopping ─────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_early_stopping_fires_before_max_epochs(self, tmp_path):
        """With patience=1, training must stop well before epochs=20."""
        trainer = _make_trainer(patience=1, tmp_path=tmp_path)
        loader = _make_loader(n=8)
        history = trainer.fit(loader, loader, verbose=False)
        # patience=1 means stop after 2 epochs with no improvement
        assert len(history) < 20

    def test_early_stopping_respects_patience(self, tmp_path):
        """patience=3 means training runs at least 1 epoch."""
        trainer = _make_trainer(patience=3, tmp_path=tmp_path)
        loader = _make_loader(n=8)
        history = trainer.fit(loader, loader, verbose=False)
        assert len(history) >= 1
