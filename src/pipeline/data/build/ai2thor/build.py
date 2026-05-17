"""Build AI2THOR knowledge graph + claims in a single step.

Runs build_rdf (TTL generation) then build_claims (claim extraction) sequentially.

Usage:
    python -m src.pipeline.data.ai2thor.build \
        --config configs/config.yaml \
        --out-rdf out/knowledge_graph.ttl \
        --output-dir data/raw/ai2thor \
        --max-contexts 10 \
        --verbose
"""
from __future__ import annotations

import argparse
import sys

from src.pipeline.data.build.ai2thor.build_rdf import build_ai2thor_rdf, BuildRDFResultSummary
from src.pipeline.data.build.ai2thor.build_claims import build_claims, _load_generation_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build AI2THOR RDF knowledge graph and generate labeled claims."
    )
    ap.add_argument("--config", required=True, help="Path to ai2thor YAML config")
    ap.add_argument("--out-rdf", default="out/model/knowledge_graph.ttl", help="Output TTL path")
    ap.add_argument("--output-dir", default="data/raw/ai2thor", help="Claims output directory")
    ap.add_argument("--max-contexts", type=int, default=None, help="Limit number of scenes")
    ap.add_argument("--base-iri", default=None)
    ap.add_argument("--format", default="turtle")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logger.info("=== Step 1/2: Build RDF knowledge graph ===")
    rdf_result = build_ai2thor_rdf(
        config_path=args.config,
        out_path=args.out_rdf,
        base_iri=args.base_iri,
        fmt=args.format,
        strict=args.strict,
    )
    summary = BuildRDFResultSummary()
    summary.add_result(rdf_result)
    summary.print_summary(show_lists=args.verbose)

    if not rdf_result.success:
        logger.error("RDF build failed — aborting claims step.")
        sys.exit(1)

    logger.info("=== Step 2/2: Generate claims from TTL ===")
    cfg = _load_generation_config(args.config)
    claims_result = build_claims(
        ttl_path=args.out_rdf,
        output_dir=args.output_dir,
        max_contexts=args.max_contexts if args.max_contexts is not None else cfg["max_contexts"],
        n_one_hop=cfg["n_one_hop"],
        n_conjunction=cfg["n_conjunction"],
        n_negation=cfg["n_negation"],
        n_absence=cfg["n_absence"],
        add_corruption=cfg["add_corruption"],
        verbose=args.verbose,
    )
    logger.info(
        "Claims complete — %d contexts, %d triples",
        len(claims_result.contexts),
        claims_result.total_triples,
    )


if __name__ == "__main__":
    main()
