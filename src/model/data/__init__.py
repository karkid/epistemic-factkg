"""Data preparation — JSONL → PyG HeteroData graphs.

Modules:
- featurizer.py  : Sentence embeddings + categorical encodings (cached)
- builder.py     : Record → HeteroData with supervision labels
- types.py       : Node/edge constants, ClaimGraph dataclass
- dataset.py     : PyG InMemoryDataset (optional, currently bypassed)
"""
