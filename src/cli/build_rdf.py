# src/cli/build_rdf.py
from __future__ import annotations

import argparse
import json

from src.pipelines.build_rdf import build_ai2thor_rdf
from src.pipelines.result import PipelineRunResult, PipelineSummary


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
        summary = PipelineSummary()
        summary.add_result(res)
        summary.print_summary(logger_func=print, show_lists=args.verbose)



if __name__ == "__main__":
    main()
