"""SymbolicAggregator — stateless EC formula, no trainable parameters.

EC_i  = 1 - (1 - ST_i)^(EW_i * IS_i)
SupportScore = 1 - prod(1 - EC_i)  over supporters
RefuteScore  = 1 - prod(1 - EC_i)  over refuters

Called only at inference — gradients never flow through here.
"""

from __future__ import annotations

import torch


_SUPPORT_STANCE = 0   # STANCE_TO_INT["supports"] == STANCE_TO_INT["absent"]
_REFUTE_STANCE  = 1


class SymbolicAggregator:
    """Stateless: no __init__ parameters, no nn.Module inheritance."""

    def compute_soft_scores(
        self,
        stance_probs: torch.Tensor,  # [N_ev, 3] — softmax from H1 (differentiable)
        is_pred:      torch.Tensor,  # [N_ev] or [N_ev,1]
        ew:           torch.Tensor,  # [N_ev]
        st:           torch.Tensor,  # [N_ev]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable soft symbolic scores for end-to-end training.

        Uses softmax stance probabilities instead of argmax so gradients
        flow back through H1 and H2 into the encoder.

        soft_support = 1 - ∏(1 - EC_i * p_support_i)
        soft_refute  = 1 - ∏(1 - EC_i * p_refute_i)
        """
        is_flat = is_pred.view(-1).float()
        ec = 1.0 - (1.0 - st.float()) ** (ew.float() * is_flat)

        p_support = stance_probs[:, 0]
        p_refute  = stance_probs[:, 1]

        support_score = 1.0 - torch.prod(1.0 - ec * p_support)
        refute_score  = 1.0 - torch.prod(1.0 - ec * p_refute)

        return support_score.unsqueeze(0), refute_score.unsqueeze(0)  # [1], [1]

    def compute_scores(
        self,
        stance_pred: torch.Tensor,   # [N_ev] int — argmax of H1 logits
        is_pred:     torch.Tensor,   # [N_ev] or [N_ev,1] float from H2
        ew:          torch.Tensor,   # [N_ev] float — pre-computed epistemic weight
        st:          torch.Tensor,   # [N_ev] float — source trust from registry
    ) -> tuple[float, float]:
        """Return (support_score, refute_score) ∈ [0.0, 1.0]."""
        is_flat = is_pred.view(-1).float()
        ew_flat = ew.view(-1).float()
        st_flat = st.view(-1).float()

        ec = 1.0 - (1.0 - st_flat) ** (ew_flat * is_flat)

        support_mask = stance_pred == _SUPPORT_STANCE
        refute_mask  = stance_pred == _REFUTE_STANCE

        support_score = _aggregate(ec, support_mask)
        refute_score  = _aggregate(ec, refute_mask)

        return float(support_score), float(refute_score)

    def get_verdict(self, support_score: float, refute_score: float) -> str:
        if support_score >= 0.75 and refute_score < 0.40:
            return "supported"
        if refute_score >= 0.75 and support_score < 0.40:
            return "refuted"
        if support_score >= 0.40 and refute_score >= 0.40:
            return "conflicting_evidence"
        return "not_enough_evidence"


def _aggregate(ec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = ec[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.prod(1.0 - selected)
