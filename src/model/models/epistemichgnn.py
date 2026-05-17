"""EpistemicHGNN — neuro-symbolic fact-verification model (ADR-013, ADR-014).

Architecture:
  EpistemicEncoder  (shared HeteroConv, config-driven)
      ↓ evidence embeddings [N_ev, hidden_dim]
  StanceHead   H1  → stance logits  [N_ev, 3]
  ISHead       H2  → IS scalars     [N_ev, 1]
      ↓ differentiable soft EC aggregation (per claim)
  VerdictHead      → verdict logits [N_claims, 3]

Training loss:
  stance_CE + λ₁ * IS_MSE + λ₂ * verdict_CE
  Gradients flow through H1 (soft stance probs) and H2 (IS) into the encoder.

At inference: hard argmax stance → symbolic EC scores → VerdictHead → verdict string.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.aggregator import SymbolicAggregator
from src.model.config import GraphConfig
from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import ISHead, StanceHead, VerdictHead
from src.model.data.types import VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


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
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim)
        self.is_head = ISHead(hidden_dim)
        self.verdict_head = VerdictHead()
        self.aggregator = SymbolicAggregator()

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            stance_logits  : [N_ev, 3]
            is_pred        : [N_ev, 1]
            ec_scores      : [N_claims, 3]  — (sup, ref, nei) soft EC scores
            verdict_logits : [N_claims, 3]  — from soft symbolic scores
        """
        x_dict = self.encoder(data)
        ev_emb = x_dict[NodeType.EVIDENCE]

        stance_logits = self.stance_head(ev_emb)  # [N_ev, 3]
        is_pred = self.is_head(ev_emb)  # [N_ev, 1]

        # Soft symbolic scores — differentiable (uses softmax probs, not argmax)
        ec_scores, verdict_logits = self._soft_verdict_logits(data, stance_logits, is_pred.detach())

        return {
            "stance_logits": stance_logits,
            "is_pred": is_pred,
            "ec_scores": ec_scores,
            "verdict_logits": verdict_logits,
        }

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Full neuro-symbolic inference for a single graph.

        Uses soft EC scores from forward (same path as training) for thresholds,
        eliminating the train-inference gap from hard-argmax EC computation.
        """
        _EC_DECISIVE = 0.35

        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)

        ec = out["ec_scores"][0]  # [3] — (sup, ref, nei) for single claim
        sup = float(ec[0])
        ref = float(ec[1])

        if ref > sup and ref > _EC_DECISIVE:
            verdict = "refuted"
        elif sup > ref and sup > _EC_DECISIVE:
            verdict = "supported"
        else:
            # EC is ambiguous — let the trained VerdictHead decide.
            # This covers both genuine NEI and neutral-stance collapse.
            verdict_idx = out["verdict_logits"].argmax(dim=-1).item()
            verdict = _INT_TO_VERDICT.get(int(verdict_idx), "not_enough_evidence")

        return {
            **out,
            "stance_pred": stance_pred,
            "support_score": sup,
            "refute_score": ref,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------

    def _soft_verdict_logits(
        self,
        data: HeteroData,
        stance_logits: torch.Tensor,
        is_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-claim soft symbolic scores and map through VerdictHead.

        Returns (ec_scores [N_claims, 3], verdict_logits [N_claims, 3]).
        Uses the batch pointer to group evidence by claim; falls back to a
        single claim when batch pointer is absent (single-graph inference).
        """
        ev = data[NodeType.EVIDENCE]
        batch_ptr = getattr(ev, "batch", None)
        n_claims = data[NodeType.CLAIM].x.shape[0]

        if batch_ptr is None:
            batch_ptr = torch.zeros(
                stance_logits.shape[0], dtype=torch.long, device=stance_logits.device
            )

        stance_probs = torch.softmax(stance_logits, dim=-1)
        scores = torch.zeros(n_claims, 3, device=stance_logits.device)

        for c in range(n_claims):
            mask = batch_ptr == c
            sup, ref, nei = self.aggregator.compute_soft_scores(
                stance_probs[mask],
                is_pred[mask],
                ew=ev.ew[mask],
                st=ev.st[mask],
            )
            scores[c, 0] = sup
            scores[c, 1] = ref
            scores[c, 2] = nei

        return scores, self.verdict_head(scores)  # [n_claims, 3], [n_claims, 3]
