"""Training loop for EpistemicHGNN V1 (neuro-symbolic, ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.core.gnn.model import EpistemicHGNN
from src.core.gnn.types import NodeType


@dataclass
class TrainConfig:
    epochs:         int   = 50
    lr:             float = 1e-3
    batch_size:     int   = 32
    dropout:        float = 0.3
    hidden_dim:     int   = 256
    heads:          int   = 4
    is_loss_weight: float = 0.5   # λ: total = stance_loss + λ * is_loss
    device:         str   = "cpu"
    checkpoint_dir: str   = "out/checkpoints"
    patience:       int   = 10


@dataclass
class EpochResult:
    loss:        float
    stance_loss: float
    is_loss:     float
    stance_acc:  float   # fraction of evidence items with correct stance


class Trainer:
    """Trains EpistemicHGNN with H1 (stance) + H2 (IS regression) losses.

    No verdict loss — verdict emerges from symbolic aggregation at inference.
    """

    def __init__(
        self,
        model: EpistemicHGNN,
        config: TrainConfig,
        stance_class_weights: torch.Tensor | None = None,
    ):
        self.model  = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.stance_criterion = nn.CrossEntropyLoss(
            weight=stance_class_weights.to(self.device) if stance_class_weights is not None else None
        )
        self.is_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        self._best_val_loss = float("inf")
        self._epochs_no_improve = 0
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = ckpt_dir / "best_model.pt"

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        self.model.train(train)
        total_loss = total_stance = total_is = correct_stance = n_items = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device)
                out   = self.model(batch)

                stance_y = batch[NodeType.EVIDENCE].stance_y.view(-1)   # [N_ev]
                is_y     = batch[NodeType.EVIDENCE].is_y.view(-1, 1)     # [N_ev, 1]

                s_loss = self.stance_criterion(out["stance_logits"], stance_y)
                i_loss = self.is_criterion(out["is_pred"], is_y)
                loss   = s_loss + self.config.is_loss_weight * i_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                n = stance_y.size(0)
                total_loss   += loss.item()    * n
                total_stance += s_loss.item()  * n
                total_is     += i_loss.item()  * n
                preds = out["stance_logits"].argmax(dim=-1)
                correct_stance += (preds == stance_y).sum().item()
                n_items += n

        d = max(n_items, 1)
        return EpochResult(
            loss=total_loss / d,
            stance_loss=total_stance / d,
            is_loss=total_is / d,
            stance_acc=correct_stance / d,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        verbose: bool = True,
    ) -> list[dict]:
        history = []
        for epoch in range(1, self.config.epochs + 1):
            tr  = self._run_epoch(train_loader, train=True)
            val = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(val.loss)

            history.append({
                "epoch":       epoch,
                "train_loss":  round(tr.loss,  4),
                "train_s":     round(tr.stance_loss, 4),
                "train_is":    round(tr.is_loss, 4),
                "train_acc":   round(tr.stance_acc, 4),
                "val_loss":    round(val.loss,  4),
                "val_s":       round(val.stance_loss, 4),
                "val_is":      round(val.is_loss, 4),
                "val_acc":     round(val.stance_acc, 4),
            })

            if verbose:
                print(
                    f"Epoch {epoch:3d} | "
                    f"loss {tr.loss:.4f} (s={tr.stance_loss:.4f} is={tr.is_loss:.4f}) "
                    f"s_acc {tr.stance_acc:.3f} | "
                    f"val loss {val.loss:.4f} s_acc {val.stance_acc:.3f}"
                )

            if val.loss < self._best_val_loss:
                self._best_val_loss    = val.loss
                self._epochs_no_improve = 0
                torch.save(self.model.state_dict(), self._ckpt_path)
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return history

    def load_best(self) -> None:
        self.model.load_state_dict(
            torch.load(self._ckpt_path, map_location=self.device)
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochResult:
        return self._run_epoch(loader, train=False)
