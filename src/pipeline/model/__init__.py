"""Model pipeline stages — graph construction, training, and evaluation.

Stages:
1. build_graphs.py  — Convert filtered JSONL → PyG HeteroData graphs (out/graphs/)
2. train.py         — Train EpistemicHGNN on train/val splits (out/checkpoints/)
3. evaluate.py      — Evaluate on test split (out/evaluation/)
4. report.py        — Generate summary statistics

Run after data/ pipeline is complete. Can run train.py multiple times with different
hyperparameters (different --hidden-dim, --is-loss-weight, etc.) without re-running data/.
"""
