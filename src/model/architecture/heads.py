"""Stance and Inference-Strength prediction heads.

Both heads are graph-structure-agnostic — they only see evidence embeddings
from the encoder output. They never know or care what node types exist.

H1 StanceHead  : 3-class classification (supports / refutes / neutral)
H2 ISHead      : scalar regression in [0, 1] (inference strength)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StanceHead(nn.Module):
    """H1 — per-evidence stance classification.

    Input : evidence embeddings [N_ev, hidden_dim]
    Output: logits [N_ev, 3]  (0=supports, 1=refutes, 2=neutral)
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, ev_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(ev_emb)


class VerdictHead(nn.Module):
    """Learned verdict calibration: maps (support_score, refute_score) → 3-class logits.

    Replaces hard-coded thresholds (0.75 / 0.40) with a learned Linear(2→3).
    Trained with claim-level CrossEntropyLoss against annotated verdicts.
    The EC formula (symbolic aggregation) is unchanged — this head only
    learns where the decision boundaries sit for each dataset's annotation style.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """scores: [N_claims, 2] — (support_score, refute_score) per claim."""
        return self.linear(scores)


class HybridVerdictHead(nn.Module):
    """Verdict head that fuses symbolic EC scores with the claim node embedding.

    Input : EC scores [N_claims, 2] + claim embedding [N_claims, hidden_dim]
    Output: logits [N_claims, 3]

    The fusion lets the head use both the interpretable epistemic signal
    (support_score / refute_score from the EC formula) and the full semantic
    context from the encoder, so neither source of information is discarded.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, scores: torch.Tensor, claim_emb: torch.Tensor) -> torch.Tensor:
        """scores: [N_claims, 2], claim_emb: [N_claims, hidden_dim]."""
        return self.mlp(torch.cat([scores, claim_emb], dim=1))


class ISHead(nn.Module):
    """H2 — per-evidence inference-strength regression.

    Input : evidence embeddings [N_ev, hidden_dim]
    Output: IS scalars [N_ev, 1] in [0, 1] (via Sigmoid)
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, ev_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(ev_emb)
