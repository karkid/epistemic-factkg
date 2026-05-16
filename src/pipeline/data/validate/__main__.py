"""Entry point: python -m src.pipeline.data.validate <subcommand> [args]

Subcommands:
  synthetic  — validate synthetic JSONL for shortcut-breaking requirements
  unified    — validate unified v3.0 JSONL against schema + dataset rules
  training   — validate training JSONL against ADR-012 distribution targets
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate pipeline data at different stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    syn = sub.add_parser("synthetic", help="Validate synthetic JSONL batch.")
    syn.add_argument("--input", required=True)
    syn.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")

    uni = sub.add_parser("unified", help="Validate unified v3.0 JSONL.")
    uni.add_argument("--files", nargs="+", required=True)
    uni.add_argument("--out", required=True)
    uni.add_argument("--schema", default=None)

    trn = sub.add_parser("training", help="Validate training JSONL vs ADR-012 targets.")
    trn.add_argument("--input", required=True)
    trn.add_argument("--config", default="configs/config.yaml")
    trn.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "synthetic":
        from src.pipeline.data.validate.validate_synthetic import run

        sys.exit(run(args))
    elif args.cmd == "unified":
        from src.pipeline.data.validate.validate_unified import run

        sys.exit(run(args))
    elif args.cmd == "training":
        from src.pipeline.data.validate.validate_training import run

        sys.exit(run(args))


if __name__ == "__main__":
    main()
