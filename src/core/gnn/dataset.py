"""PyG dataset that loads the filtered training JSONL and builds ClaimGraph objects."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from src.core.claims.labels import load_source_trust_registry
from src.core.gnn.featurizer import Featurizer
from src.core.gnn.graph_builder import ClaimGraphBuilder
from src.core.gnn.types import NUM_VERDICT


class EpistemicFactDataset(InMemoryDataset):
    """Loads epistemic_factkg_training.jsonl (post ADR-011 + ADR-015 filter) into memory.

    Serialises built graphs to a .pt cache file so subsequent loads skip graph construction.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        pt_cache: str | Path,
        registry_path: str | Path = "data/registry/source_trust_registry.jsonl",
        featurizer: Featurizer | None = None,
        force_rebuild: bool = False,
    ):
        self._jsonl_path = Path(jsonl_path)
        self._pt_cache = Path(pt_cache)
        self._registry = load_source_trust_registry(registry_path)
        self._featurizer = featurizer or Featurizer()
        self._force_rebuild = force_rebuild

        # InMemoryDataset.__init__ triggers download/process if needed
        super().__init__(root=str(self._pt_cache.parent))

        self.load(str(self._pt_cache))

    @property
    def processed_file_names(self) -> list[str]:
        return [self._pt_cache.name]

    def process(self) -> None:
        builder = ClaimGraphBuilder(self._registry, self._featurizer)
        graphs = []

        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cg = builder.build(record)
                if cg.label < 0:
                    continue  # skip unmapped verdicts (e.g. conflicting_evidence if any slip through)

                # Attach metadata as graph-level tensors
                cg.data.y = torch.tensor([cg.label], dtype=torch.long)
                cg.data.dataset_name = cg.dataset
                graphs.append(cg.data)

        self._featurizer.save_cache()
        self.save(graphs, str(self._pt_cache))

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted CrossEntropyLoss (3-class, ADR-015)."""
        all_y = self._data.y.view(-1)
        counts = torch.bincount(all_y, minlength=NUM_VERDICT).float().clamp(min=1.0)
        return counts.sum() / (NUM_VERDICT * counts)
