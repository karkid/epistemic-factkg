"""TrainConfig — hyperparameters for EpistemicHGNN training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.3
    hidden_dim: int = 256
    heads: int = 4
    is_loss_weight: float = 0.5  # λ₁: stance + λ₁*is + λ₂*verdict
    verdict_loss_weight: float = 1.0  # λ₂
    device: str = "cpu"
    checkpoint_dir: str = "out/model/checkpoints"
    report_dir: str = "out/reports/model"
    patience: int = 10
