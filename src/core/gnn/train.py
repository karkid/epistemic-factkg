"""Training loop for EpistemicHGNN (ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.core.gnn.model import EpistemicHGNN


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.3
    hidden_dim: int = 256
    heads: int = 2
    use_modality_learning: bool = False
    aux_loss_weight: float = 0.0  # weight for Pramana aux head loss (Pathway B)
    masked_edge_types: list[str] = field(default_factory=list)  # edge relations to zero (ADR-016)
    device: str = "cpu"
    checkpoint_dir: str = "out/checkpoints"
    patience: int = 10  # early stopping — epochs without val improvement


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    pramana_loss: float = 0.0


class Trainer:
    """Trains EpistemicHGNN with weighted CrossEntropyLoss and early stopping."""

    def __init__(
        self,
        model: EpistemicHGNN,
        class_weights: torch.Tensor | None,
        config: TrainConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )

        self._best_val_acc = 0.0
        self._epochs_no_improve = 0
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = checkpoint_dir / "best_model.pt"

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        self.model.train(train)
        total_loss = total_pramana_loss = correct = total = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device)

                # Zero masked edge types in-place (ADR-016 stance-removal ablation)
                for rel in self.config.masked_edge_types:
                    for et in batch.edge_types:
                        if et[1] == rel:
                            batch[et].edge_index = torch.zeros(
                                (2, 0), dtype=torch.long, device=self.device
                            )

                out = self.model(batch)

                labels = batch["claim"].y if hasattr(batch["claim"], "y") else batch.y
                labels = labels.view(-1)

                loss = self.criterion(out["verdict"], labels)

                if (
                    self.config.use_modality_learning
                    and "pramana" in out
                    and self.config.aux_loss_weight > 0
                ):
                    pramana_labels = batch["claim"].pramana_y.view(-1)
                    pramana_loss = nn.CrossEntropyLoss()(out["pramana"], pramana_labels)
                    total_pramana_loss += pramana_loss.item()
                    loss = loss + self.config.aux_loss_weight * pramana_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = out["verdict"].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return EpochResult(
            loss=total_loss / max(total, 1),
            accuracy=correct / max(total, 1),
            pramana_loss=total_pramana_loss / max(total, 1),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> list[dict]:
        history = []
        for epoch in range(1, self.config.epochs + 1):
            train_res = self._run_epoch(train_loader, train=True)
            val_res = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_res.accuracy)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_res.loss, 4),
                    "train_acc": round(train_res.accuracy, 4),
                    "val_loss": round(val_res.loss, 4),
                    "val_acc": round(val_res.accuracy, 4),
                }
            )

            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train loss {train_res.loss:.4f} acc {train_res.accuracy:.3f} | "
                    f"val loss {val_res.loss:.4f} acc {val_res.accuracy:.3f}"
                )

            if val_res.accuracy > self._best_val_acc:
                self._best_val_acc = val_res.accuracy
                self._epochs_no_improve = 0
                torch.save(self.model.state_dict(), self._ckpt_path)
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.config.patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch} (no val improvement for {self.config.patience} epochs)"
                        )
                    break

        return history

    def load_best(self) -> None:
        self.model.load_state_dict(
            torch.load(self._ckpt_path, map_location=self.device)
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochResult:
        return self._run_epoch(loader, train=False)
