"""BaselineHGNN — ablation model for publication comparison.

Same graph encoder and multi-task supervision as EpistemicHGNN, but the
verdict is predicted directly from claim node embeddings after message
passing — bypassing the Pramana EC formula and symbolic aggregation.

Ablation story (for paper):
  EpistemicHGNN:  evidence → IS → EC formula → SymbolicAggregator → VerdictHead
  BaselineHGNN:   evidence → HeteroConv → claim node → MLP → verdict

Everything upstream of verdict is identical: same graph structure, same
HeteroConv encoder, same StanceHead (H1), same ISHead (H2).  The only
difference is the verdict pathway — no EC weighting, no symbolic scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import ISHead, StanceHead
from src.model.config import GraphConfig
from src.model.data.types import NUM_VERDICT, VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


class BaselineHGNN(nn.Module):
    """Ablation baseline: HeteroConv encoder + direct claim-node verdict.

    Args:
        graph_config:  Node dims + edge types (use GraphConfig.v1()).
        hidden_dim:    Encoder output dim; shared with stance and IS heads.
        heads:         GAT attention heads in encoder.
        dropout:       Encoder inter-layer and verdict MLP dropout.
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
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim)
        self.is_head = ISHead(hidden_dim)
        # Verdict from claim node only — no EC formula, no symbolic aggregation
        self.verdict_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_VERDICT),
        )

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            stance_logits  : [N_ev, 3]
            is_pred        : [N_ev, 1]
            verdict_logits : [N_claims, 3]  — from claim node MLP, no EC
        """
        x_dict = self.encoder(data)
        ev_emb = x_dict[NodeType.EVIDENCE]  # [N_ev, hidden_dim]
        claim_emb = x_dict[NodeType.CLAIM]  # [N_claims, hidden_dim]

        return {
            "stance_logits": self.stance_head(ev_emb),
            "is_pred": self.is_head(ev_emb),
            "verdict_logits": self.verdict_mlp(claim_emb),
        }

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Inference — no symbolic scores (baseline has no EC formula)."""
        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)
        verdict_idx = out["verdict_logits"].argmax(dim=-1).item()
        verdict = _INT_TO_VERDICT.get(int(verdict_idx), "not_enough_evidence")
        return {
            **out,
            "stance_pred": stance_pred,
            "verdict": verdict,
        }
