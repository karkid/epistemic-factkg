"""NLIHybridHGNN — v3-nli: HybridHGNN with NLI stance probs used directly in EC formula.

Evidence node features are 403d — a frozen NLI cross-encoder (DeBERTa-v3-small, MNLI)
contributes 3 stance probability columns stored at ev.x[:, -3:]:
  [p_contradiction, p_entailment, p_neutral]

Key difference from v2-hgnn: _soft_verdict_logits bypasses H1's learned stance probs
and feeds the frozen NLI probs directly into the EC formula. H1 still trains on its
own stance CE loss, but the verdict gradient path uses the cross-encoder signal.

NLI → EC column mapping:
  p_entailment   (col 1) → p_supports  (col 0 in EC formula)
  p_contradiction (col 0) → p_refutes  (col 1 in EC formula)
  p_neutral       (col 2) → p_neutral  (col 2 in EC formula)
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.model.config import GraphConfig
from src.model.data.types import NodeType
from src.model.models.hybridhgnn import HybridHGNN


class NLIHybridHGNN(HybridHGNN):
    """v3-nli: HybridHGNN where the EC formula uses frozen NLI probs, not H1.

    Args:
        graph_config: Defaults to GraphConfig.v2() (403d evidence).
        hidden_dim:   Encoder output dim — same as v2-hgnn (256).
        heads:        GAT attention heads — same as v2-hgnn (4).
        dropout:      Dropout — same as v2-hgnn (0.1).
    """

    def __init__(
        self,
        graph_config: GraphConfig | None = None,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        cfg = graph_config or GraphConfig.v2()
        super().__init__(cfg, hidden_dim, heads, dropout)

    def _soft_verdict_logits(
        self,
        data: HeteroData,
        stance_logits: torch.Tensor,
        is_pred: torch.Tensor,
        claim_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EC formula using frozen NLI probs instead of H1's learned stance probs.

        ev.x[:, -3:] = [p_contradiction, p_entailment, p_neutral] (already softmaxed).
        EC formula expects  [p_supports,   p_refutes,    p_neutral].
        Reorder: columns [1, 0, 2] — entailment→supports, contradiction→refutes.

        Returns (ec_scores [N_claims, 3], verdict_logits [N_claims, 3]).
        """
        ev = data[NodeType.EVIDENCE]
        batch_ptr = getattr(ev, "batch", None)
        n_claims = data[NodeType.CLAIM].x.shape[0]

        if batch_ptr is None:
            batch_ptr = torch.zeros(
                stance_logits.shape[0], dtype=torch.long, device=stance_logits.device
            )

        # [N_ev, 3]: reorder NLI cols to match EC formula's [supports, refutes, neutral]
        nli_probs = ev.x[:, -3:]                # [p_contra, p_entail, p_neutral]
        stance_probs = nli_probs[:, [1, 0, 2]]  # [p_supports, p_refutes, p_neutral]

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

        return scores, self.verdict_head(scores, claim_emb)

    @torch.no_grad()
    def predict(self, data: HeteroData) -> dict:
        """Inference using NLI EC scores for verdict when signal is decisive.

        Uses soft NLI-derived EC scores from forward (same path as training)
        for threshold checks — no train-inference gap.
        """
        _EC_DECISIVE = 0.35

        from src.model.data.types import VERDICT_TO_INT
        _int_to_verdict = {v: k for k, v in VERDICT_TO_INT.items()}

        out = self.forward(data)
        ev = data[NodeType.EVIDENCE]

        # stance_pred from NLI argmax (for interpretability display only)
        nli_probs = ev.x[:, -3:]
        stance_pred = nli_probs[:, [1, 0, 2]].argmax(dim=-1)  # [N_ev]

        # Use soft EC from forward — consistent with training
        ec = out["ec_scores"][0]  # [3] — (sup, ref, nei) for single claim
        sup = float(ec[0])
        ref = float(ec[1])

        if ref > sup and ref > _EC_DECISIVE:
            verdict = "refuted"
        elif sup > ref and sup > _EC_DECISIVE:
            verdict = "supported"
        else:
            # EC is ambiguous — let the trained VerdictHead decide.
            verdict = _int_to_verdict[int(out["verdict_logits"].argmax(dim=-1).item())]

        return {
            **out,
            "stance_pred": stance_pred,
            "support_score": sup,
            "refute_score": ref,
            "verdict": verdict,
        }
