"""CLI: validate a synthetic JSONL batch for shortcut-breaking and structural requirements.

Usage:
    python -m src.cli.validate_synthetic \\
        --input data/raw/synthetic/batch_001.jsonl \\
        --registry data/registry/source_trust_registry.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Validate a synthetic data batch.")
    ap.add_argument("--input", required=True, help="Synthetic JSONL file to validate.")
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    args = ap.parse_args()

    from src.adapters.synthetic.validator import validate_file
    report = validate_file(args.input, args.registry)
    print(report.summary())

    sys.exit(0 if report.passes else 1)


if __name__ == "__main__":
    main()
