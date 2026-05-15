"""CLI: generate synthetic shortcut-breaking claims.

Usage:
    # Offline (no API key needed):
    python -m src.cli.generate_synthetic --output data/raw/synthetic/batch_001.jsonl

    # Grounded (uses seed pool):
    python -m src.cli.generate_synthetic --client grounded --output ...

    # LLM (requires ANTHROPIC_API_KEY):
    python -m src.cli.generate_synthetic --client llm --output ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic epistemic claims.")
    ap.add_argument("--config",    default="configs/config.yaml")
    ap.add_argument("--registry",  default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--seed-pool", default="data/registry/seed_pool.jsonl",
                    help="Seed pool JSONL for grounded client.")
    ap.add_argument("--ai2thor-claims", default="data/raw/ai2thor/claims_all.jsonl",
                    help="AI2THOR claims JSONL for perception/non_apprehension triplets.")
    ap.add_argument("--n-records", type=int, default=None,
                    help="Total records to generate (overrides config).")
    ap.add_argument("--output",    required=True, help="Output JSONL path.")
    ap.add_argument("--client",    default=None,
                    choices=["local", "grounded", "llm"],
                    help="Text client: local (offline), grounded (seed pool), llm (API). "
                         "Default: llm if ANTHROPIC_API_KEY set, grounded if seed pool exists, "
                         "else local.")
    ap.add_argument("--api-key",   default=None,
                    help="Anthropic API key (default: ANTHROPIC_API_KEY env var).")
    args = ap.parse_args()

    # Load config
    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = (raw or {}).get("synthetic", {})

    model        = cfg.get("model", "claude-haiku-4-5-20251001")
    n_records    = args.n_records or cfg.get("n_records", 100)
    distribution = cfg.get("distribution") or None

    # Load registry
    from src.core.claims.labels import load_source_trust_registry
    registry: dict = {}
    if Path(args.registry).exists():
        registry = load_source_trust_registry(args.registry)
    else:
        print(f"Warning: registry not found at {args.registry}, using defaults.", file=sys.stderr)

    # Resolve client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    from src.adapters.synthetic.client import LocalTextClient, GroundedClient
    from src.adapters.synthetic.llm import LLMClient
    from src.adapters.synthetic.fictional_generator import FictionalClaimGenerator

    client_mode = args.client
    if client_mode is None:
        if api_key:
            client_mode = "llm"
        elif Path(args.seed_pool).exists():
            client_mode = "grounded"
        else:
            client_mode = "local"

    if client_mode == "llm":
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Use --api-key or set the env var.", file=sys.stderr)
            sys.exit(1)
        client = LLMClient(model=model, api_key=api_key)
        print(f"Generating {n_records} records (client=llm, model={model})…")
    elif client_mode == "grounded":
        client = GroundedClient(
            seed_pool_path=args.seed_pool,
            ai2thor_path=args.ai2thor_claims,
        )
        print(f"Generating {n_records} records (client=grounded, seed={args.seed_pool})…")
    else:
        client = LocalTextClient()
        print(f"Generating {n_records} records (client=local, offline)…")

    gen = FictionalClaimGenerator(registry=registry, _client=client)
    records = gen.generate_batch(n_records=n_records, distribution=distribution)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    shortcut = sum(1 for r in records if r.get("meta", {}).get("is_shortcut_breaking"))
    print(f"Wrote {len(records)} records → {out_path}")
    if records:
        print(f"Shortcut-breaking: {shortcut} ({shortcut/len(records):.1%})")


if __name__ == "__main__":
    main()
