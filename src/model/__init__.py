"""Neuro-symbolic fact verification model stack (V1, extensible to V2+).

Subpackages:
- data/          : featurizer, graph builder, dataset utilities
- architecture/  : encoder, heads (H1, H2), aggregator (EC formula ops)
- training/      : TrainConfig, Trainer (multi-task loss, checkpointing)
- evaluation/    : metrics, inference wrapper
- v1/            : V1-specific documentation (node dims, edge types)
- config.py      : GraphConfig (single source of truth for architecture schema)
- models/        : model registry (MODELS dict) + one file per model class

Design principle: Never hardcode node/edge names in architecture code.
Everything is driven by GraphConfig — V2 extensions only need config + adapter changes.
"""
