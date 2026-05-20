"""TrainConfig — hyperparameters for EpistemicHGNN training."""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    import torch  # local import — avoids torch import at module level

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    dropout: float = 0.3
    hidden_dim: int = 256
    heads: int = 4
    is_loss_weight: float = 0.5  # λ₁: stance + λ₁*is + λ₂*verdict
    verdict_loss_weight: float = 1.0  # λ₂
    device: str = field(default_factory=_default_device)  # auto: cuda > mps > cpu
    checkpoint_dir: str = "out/model/checkpoints"
    report_dir: str = "out/reports/model"
    patience: int = 5
    ec_threshold: float = 0.35
