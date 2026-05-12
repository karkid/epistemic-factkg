"""EpistemicHGNN — HeteroConv + GATConv for epistemic fact verification (ADR-013)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear

from src.core.gnn.types import NUM_PRAMANA, NUM_VERDICT


class EpistemicHGNN(nn.Module):
    """Heterogeneous GNN using GATConv per edge type.

    Architecture (ADR-013):
      1. Per-type linear projections: variable input dim → hidden_dim
      2. Layer 1: HeteroConv(GATConv per edge type)  [1-hop: claim ↔ evidence]
      3. ReLU + Dropout
      4. Layer 2: HeteroConv(GATConv per edge type)  [2-hop: claim → evidence → triple]
      5. ReLU + Dropout
      6. Verdict classifier: claim node embedding → 3 logits  (ADR-015: 3-class)

      [Pathway B only]
      7. Epistemic aux head: epistemic node embedding → 5 logits  (Pramana prediction)

    Args:
        hidden_dim:            Size of all intermediate node embeddings (default 256).
        heads:                 Number of GAT attention heads (default 2).
        dropout:               Dropout probability (default 0.3).
        use_modality_learning: If True, activate the auxiliary Pramana prediction head
                               (Pathway B — Phase 5 ablation). Default False (Pathway A).
    """

    # Input dims match ADR-014 feature vectors
    _INPUT_DIMS: dict[str, int] = {
        "claim": 384,
        "evidence": 389,  # 384 + 5 (modality one-hot)
        "epistemic": 6,  # 5 (pramana one-hot) + 1 (confidence_weight)
        "triple": 384,
    }

    def __init__(
        self,
        hidden_dim: int = 256,
        heads: int = 2,
        dropout: float = 0.3,
        use_modality_learning: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_modality_learning = use_modality_learning

        # Linear projections: variable input dim → hidden_dim (one per node type)
        self.projections = nn.ModuleDict(
            {nt: Linear(dim, hidden_dim) for nt, dim in self._INPUT_DIMS.items()}
        )

        # GATConv factory — concat=False averages across heads → output stays at hidden_dim
        def gat(edge_dim: int | None = None) -> GATConv:
            return GATConv(
                in_channels=(-1, -1),
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=False,
            )

        self.conv1 = HeteroConv(
            {
                ("claim", "has_evidence", "evidence"): gat(),
                ("evidence", "supports", "claim"): gat(),
                ("evidence", "refutes", "claim"): gat(),
                ("evidence", "absent", "claim"): gat(),
                ("evidence", "no_evidence", "claim"): gat(),
                ("claim", "has_epistemic", "epistemic"): gat(edge_dim=1),
                ("claim", "has_triple", "triple"): gat(),
                ("evidence", "from_triple", "triple"): gat(),
            },
            aggr="mean",
        )

        self.conv2 = HeteroConv(
            {
                ("claim", "has_evidence", "evidence"): gat(),
                ("evidence", "supports", "claim"): gat(),
                ("evidence", "refutes", "claim"): gat(),
                ("evidence", "absent", "claim"): gat(),
                ("evidence", "no_evidence", "claim"): gat(),
                ("claim", "has_epistemic", "epistemic"): gat(edge_dim=1),
                ("claim", "has_triple", "triple"): gat(),
                ("evidence", "from_triple", "triple"): gat(),
            },
            aggr="mean",
        )

        # Verdict classifier applied to claim node embedding
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, NUM_VERDICT),
        )

        # Auxiliary Pramana prediction head (Pathway B — Phase 5 ablation)
        if use_modality_learning:
            self.pramana_head = nn.Linear(hidden_dim, NUM_PRAMANA)

    def forward(self, data) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG HeteroData batch (or single graph).

        Returns:
            dict with key 'verdict' (logits [B, 3]) and optionally 'pramana' (logits [B, 5]).
        """
        x_dict = {nt: data[nt].x for nt in data.node_types if nt in self.projections}

        # Project each node type to hidden_dim
        x_dict = {nt: self.projections[nt](x) for nt, x in x_dict.items()}

        # Build edge_index_dict and edge_attr_dict from available edge types
        edge_index_dict = {}
        edge_attr_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                edge_attr_dict[et] = data[et].edge_attr

        # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {
            k: F.dropout(v, p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {
            k: F.dropout(v, p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }

        out: dict[str, torch.Tensor] = {}

        # Verdict logits from claim node (one per graph in the batch)
        claim_emb = x_dict["claim"]  # [B, hidden_dim] after batching
        out["verdict"] = self.classifier(claim_emb)

        # Auxiliary Pramana logits (Pathway B)
        if self.use_modality_learning and "epistemic" in x_dict:
            out["pramana"] = self.pramana_head(x_dict["epistemic"])

        return out
