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
