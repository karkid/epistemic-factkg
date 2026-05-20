"""HybridHGNN — neuro-symbolic model with fused EC + claim-embedding verdict (v2-hgnn).

Architecture is identical to EpistemicHGNN (v1-hgnn) except for the verdict pathway:

  v1-hgnn:  EC scores [2d]              → VerdictHead        → verdict
  v2-hgnn:  EC scores [2d] + claim_emb  → HybridVerdictHead  → verdict

Everything upstream is shared: same encoder, same StanceHead (H1), same ISHead (H2),
same EC formula, same IS detach so IS regression is clean.

Ablation story (for paper):
  baseline   — claim_emb only          — no epistemic formalism
  v1-hgnn    — EC scores only          — pure symbolic, information bottleneck
  v2-hgnn    — EC scores + claim_emb  — hybrid; isolates EC contribution
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.model.architecture.aggregator import SymbolicAggregator
from src.model.architecture.encoder import EpistemicEncoder
from src.model.architecture.heads import HybridVerdictHead, ISHead, StanceHead
from src.model.config import GraphConfig
from src.model.data.types import VERDICT_TO_INT, NodeType

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}


class HybridHGNN(nn.Module):
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
        ec_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        cfg = graph_config or GraphConfig.v1()
        self.encoder = EpistemicEncoder(cfg, hidden_dim, heads, dropout)
        self.stance_head = StanceHead(hidden_dim)
        self.is_head = ISHead(hidden_dim)
        self.verdict_head = HybridVerdictHead(hidden_dim)
        self.aggregator = SymbolicAggregator()
        self.ec_threshold = ec_threshold

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            stance_logits  : [N_ev, 3]
            is_pred        : [N_ev, 1]
            ec_scores      : [N_claims, 3]  — (sup, ref, nei) soft EC scores
            verdict_logits : [N_claims, 3]  — from EC scores fused with claim embedding
        """
        x_dict = self.encoder(data)
        ev_emb = x_dict[NodeType.EVIDENCE]
        claim_emb = x_dict[NodeType.CLAIM]

        stance_logits = self.stance_head(ev_emb)
        is_pred = self.is_head(ev_emb)

        ec_scores, verdict_logits = self._soft_verdict_logits(
            data, stance_logits, is_pred.detach(), claim_emb
        )

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
        _EC_DECISIVE = self.ec_threshold

        out = self.forward(data)
        stance_pred = out["stance_logits"].argmax(dim=-1)

        ec = out["ec_scores"][0]  # [3] — (sup, ref, nei) for single claim
        sup = float(ec[0])
        ref = float(ec[1])

        if sup > _EC_DECISIVE and ref > _EC_DECISIVE:
            # Both sides strong — conflicting evidence, VerdictHead decides.
            verdict_idx = out["verdict_logits"].argmax(dim=-1).item()
            verdict = _INT_TO_VERDICT.get(int(verdict_idx), "not_enough_evidence")
        elif ref > sup and ref > _EC_DECISIVE:
            verdict = "refuted"
        elif sup > ref and sup > _EC_DECISIVE:
            verdict = "supported"
        else:
            # EC weak on both sides — VerdictHead decides.
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
        claim_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft EC scores and fuse with claim embedding → VerdictHead.

        Returns (ec_scores [N_claims, 3], verdict_logits [N_claims, 3]).
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

        return scores, self.verdict_head(scores, claim_emb)  # [n_claims, 3], [n_claims, 3]
