# src/cli/build_rdf.py
from __future__ import annotations

import argparse
import json

from src.pipelines.build_rdf import build_ai2thor_rdf
from src.pipelines.result import PipelineRunResult


def print_pipeline_result(
    res: PipelineRunResult,
    *,
    show_lists: bool = False,
    max_items: int = 5,
) -> None:
    status = "OK" if res.success else "FAILED"

    print()
    print("=" * 72)
    print(f"Build RDF: {status}")
    print("=" * 72)

    print(f"Output     : {res.out_path}")
    print(f"Format     : {res.format}")
    print(f"Triples    : {res.total_triples}")
    print(f"Objects    : {res.num_objects}")
    print(f"Relations  : {res.num_relations}")

    if res.contexts_processed:
        print(f"Scenes     : {len(res.contexts_processed)}")
        sample = ", ".join(res.contexts_processed[:3])
        print(f"Scene IDs  : {sample}" + (" ..." if len(res.contexts_processed) > 3 else ""))

    print("-" * 72)
    print(f"Warnings   : {len(res.warnings)}")
    print(f"Errors     : {len(res.errors)}")

    if res.warnings:
        print("\nWarnings (sample):")
        for w in res.warnings[:max_items]:
            print(f"  - {w}")
        if len(res.warnings) > max_items:
            print(f"  ... ({len(res.warnings) - max_items} more)")

    if res.errors:
        print("\nErrors (sample):")
        for e in res.errors[:max_items]:
            print(f"  - {e}")
        if len(res.errors) > max_items:
            print(f"  ... ({len(res.errors) - max_items} more)")

    if show_lists:
        print("\nFULL WARNINGS:")
        for w in res.warnings:
            print(f"  - {w}")

        print("\nFULL ERRORS:")
        for e in res.errors:
            print(f"  - {e}")

    print("=" * 72)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RDF (Turtle) from AI2-THOR scenes.")
    ap.add_argument("--config", required=True, help="Path to ai2thor YAML config")
    ap.add_argument("--out", required=True, help="Output .ttl path")
    ap.add_argument("--base-iri", default=None, help="Override base IRI for namespaces")
    ap.add_argument("--format", default="turtle", help="rdflib serialize format (default: turtle)")
    ap.add_argument("--strict", action="store_true", help="Fail on validation errors")
    ap.add_argument("--json", action="store_true", help="Print result as JSON")
    ap.add_argument("--verbose", action="store_true", help="Print full warnings/errors")


    args = ap.parse_args()

    res:PipelineRunResult = build_ai2thor_rdf(
        config_path=args.config,
        out_path=args.out,
        base_iri=args.base_iri,
        fmt=args.format,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_pipeline_result(res, show_lists=args.verbose)



if __name__ == "__main__":
    main()
