"""Data pipeline stages — JSONL generation, filtering, and splitting.

Stages:
1. generate_synthetic.py   — Generate synthetic data
2. build_claims.py         — AI2THOR simulator claims
3. convert_to_unified.py   — Run all adapters → unified JSONL (out/unified/)
4. filter_training.py      — Remove non-training evidence types (out/training/)
5. split_dataset.py        — Train/val/test split (out/splits/)

Each stage reads from previous stage output, validates, and writes to next stage output.
Run these once per new dataset. Then run model/ pipeline.
"""
