"""Single-graph inference wrapper for EpistemicHGNN."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.model.epistemichgnn import EpistemicHGNN


def predict_single(
    model: EpistemicHGNN, graph: HeteroData, device: str = "cpu"
) -> dict:
    """Run inference on a single HeteroData graph.

    Returns the same dict as EpistemicHGNN.predict(), with the model
    moved to eval mode and restored afterward.
    """
    was_training = model.training
    model.eval()
    graph = graph.to(device)
    with torch.no_grad():
        result = model.predict(graph)
    if was_training:
        model.train()
    return result


def batch_predict(
    model: EpistemicHGNN,
    graphs: list[HeteroData],
    device: str = "cpu",
) -> list[dict]:
    """Run inference over a list of HeteroData graphs."""
    return [predict_single(model, g, device) for g in graphs]
