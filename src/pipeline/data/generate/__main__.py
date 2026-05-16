"""Entry point: python -m src.pipeline.data.generate <subcommand> [args]

Subcommands:
  synthetic  — generate synthetic shortcut-breaking claims
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate pipeline data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    syn = sub.add_parser(
        "synthetic", help="Generate synthetic shortcut-breaking claims."
    )
    syn.add_argument("--config", default="configs/config.yaml")
    syn.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    syn.add_argument("--seed-pool", default="data/registry/seed_pool.jsonl")
    syn.add_argument("--ai2thor-claims", default="data/raw/ai2thor/claims_all.jsonl")
    syn.add_argument("--n-records", type=int, default=None)
    syn.add_argument("--output", required=True)
    syn.add_argument("--client", default=None, choices=["local", "grounded", "llm"])
    syn.add_argument("--api-key", default=None)

    args = ap.parse_args()

    if args.cmd == "synthetic":
        from src.pipeline.data.generate.generate_synthetic import run

        sys.exit(run(args))


if __name__ == "__main__":
    main()
