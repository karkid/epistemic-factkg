"""EpistemicHGNN — neuro-symbolic fact-verification model.

Architecture:
  EpistemicEncoder  (shared HeteroConv, config-driven)
      ↓ evidence embeddings [N_ev, hidden_dim]
  StanceHead   H1  → stance logits  [N_ev, 3]
  ISHead       H2  → IS scalars     [N_ev, 1]
      ↓ at inference only
  SymbolicAggregator  (stateless EC formula)
      → support_score, refute_score → verdict string

Training loss: CrossEntropy(stance_logits, stance_y) + λ * MSE(is_pred, is_y)
No verdict loss — verdict emerges from symbolic aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.core.gnn.aggregator import SymbolicAggregator
from src.core.gnn.config import GraphConfig
from src.core.gnn.encoder import EpistemicEncoder
from src.core.gnn.heads import ISHead, StanceHead
from src.core.gnn.types import NodeType


class EpistemicHGNN(nn.Module):
    """
    Args:
        graph_config:  Node dims + edge types (use GraphConfig.v1()).
        hidden_dim:    Encoder output dim; heads read this dim.
        heads:         GAT attention heads in encoder.
        dropout:       Encoder inter-layer dropout.
    """

    def __init__(
        self,
        graph_config: GraphConfig | None = None,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = graph_config or GraphConfig.v1()
        self.encoder    = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim)
        self.is_head     = ISHead(hidden_dim)
        self.aggregator  = SymbolicAggregator()

    def forward(
        self, data: HeteroData
    ) -> dict[str, torch.Tensor]:
        """Forward pass — used during training.

        Returns:
            stance_logits : [N_ev, 3]
            is_pred       : [N_ev, 1]
        """
        x_dict = self.encoder(data)
        ev_emb = x_dict[NodeType.EVIDENCE]   # [N_ev, hidden_dim]

        return {
            "stance_logits": self.stance_head(ev_emb),
            "is_pred":       self.is_head(ev_emb),
        }

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Full neuro-symbolic inference — used at eval/inference time.

        Returns:
            stance_logits, is_pred (same as forward), plus:
            stance_pred   : [N_ev] int argmax
            support_score : float
            refute_score  : float
            verdict       : str
        """
        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)

        ev = data[NodeType.EVIDENCE]
        support_score, refute_score = self.aggregator.compute_scores(
            stance_pred,
            out["is_pred"],
            ew=ev.ew,
            st=ev.st,
        )
        verdict = self.aggregator.get_verdict(support_score, refute_score)

        return {
            **out,
            "stance_pred":   stance_pred,
            "support_score": support_score,
            "refute_score":  refute_score,
            "verdict":       verdict,
        }
