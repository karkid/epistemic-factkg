"""Validate synthetic JSONL against shortcut-breaking requirements."""

from __future__ import annotations

import argparse
import sys


def run(args) -> int:
    from src.adapters.synthetic.validator import validate_file

    report = validate_file(args.input, args.registry)
    print(report.summary())
    return 0 if report.passes else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate synthetic JSONL batch.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--registry", default="data/registry/source_trust_registry.jsonl")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
