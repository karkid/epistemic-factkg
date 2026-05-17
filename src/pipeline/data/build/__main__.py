"""Entry point: python -m src.pipeline.data.build

Runs the full data preparation pipeline in two steps:
  1. convert — merge all sources (AI2THOR + AVeriTeC + synthetic) into unified v3.0 JSONL
  2. filter  — remove non-training evidence types to produce the GNN training JSONL

Pass --rebuild / -r to re-generate AI2THOR claims via the simulator first.

Usage:
    python -m src.pipeline.data.build \\
        --registry data/registry/source_trust_registry.jsonl \\
        --averitec data/raw/averitec/train.json data/raw/averitec/dev.json \\
        --ai2thor  data/raw/ai2thor/claims_all.jsonl \\
        --synthetic data/raw/synthetic/synthetic_current.jsonl \\
        --unified-out out/data/unified/epistemic_factkg.jsonl \\
        --training-out out/data/training/epistemic_factkg_training.jsonl \\
        --intermediate-dir out/data/intermediate

    # Re-simulate AI2THOR first:
    python -m src.pipeline.data.build --rebuild \\
        --config configs/config.yaml \\
        --out-rdf out/model/knowledge_graph.ttl \\
        --ai2thor-dir data/raw/ai2thor \\
        ...
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


def _run_ai2thor_rebuild(args) -> None:
    from src.pipeline.data.build.ai2thor.build_rdf import build_ai2thor_rdf, BuildRDFResultSummary
    from src.pipeline.data.build.ai2thor.build_claims import build_claims, _load_generation_config

    print("=== Step 0/2: Re-generate AI2THOR claims via simulator ===")
    rdf_result = build_ai2thor_rdf(
        config_path=args.config,
        out_path=args.out_rdf,
    )
    summary = BuildRDFResultSummary()
    summary.add_result(rdf_result)
    summary.print_summary()
    if not rdf_result.success:
        print("ERROR: AI2THOR RDF build failed — aborting.", file=sys.stderr)
        sys.exit(1)

    cfg = _load_generation_config(args.config)
    build_claims(
        ttl_path=args.out_rdf,
        output_dir=args.ai2thor_dir,
        max_contexts=args.max_contexts if args.max_contexts is not None else cfg["max_contexts"],
        n_one_hop=cfg["n_one_hop"],
        n_conjunction=cfg["n_conjunction"],
        n_negation=cfg["n_negation"],
        n_absence=cfg["n_absence"],
        add_corruption=cfg["add_corruption"],
        verbose=args.verbose,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dataset: convert all sources to unified JSONL, then filter for training."
    )
    ap.add_argument("--rebuild", "-r",     action="store_true",
                    help="Re-generate AI2THOR claims via simulator before building")
    ap.add_argument("--config",            default="configs/config.yaml",
                    help="Config YAML path (required when --rebuild is set)")
    ap.add_argument("--out-rdf",           default="out/model/knowledge_graph.ttl", dest="out_rdf",
                    help="Output TTL path for AI2THOR RDF (used with --rebuild)")
    ap.add_argument("--ai2thor-dir",       default="data/raw/ai2thor", dest="ai2thor_dir",
                    help="AI2THOR raw output directory (used with --rebuild)")
    ap.add_argument("--max-contexts",      type=int, default=None, dest="max_contexts")
    ap.add_argument("--registry",          default="data/registry/source_trust_registry.jsonl")
    ap.add_argument("--averitec",          nargs="*", default=[], metavar="FILE")
    ap.add_argument("--ai2thor",           nargs="*", default=[], metavar="FILE")
    ap.add_argument("--synthetic",         nargs="*", default=[], metavar="FILE")
    ap.add_argument("--unified-out",       required=True, dest="unified_out",
                    help="Output path for merged unified JSONL")
    ap.add_argument("--training-out",      required=True, dest="training_out",
                    help="Output path for filtered training JSONL")
    ap.add_argument("--intermediate-dir",  default=None, dest="intermediate_dir")
    ap.add_argument("--verbose", "-v",     action="store_true")
    args = ap.parse_args()

    if args.rebuild:
        _run_ai2thor_rebuild(args)

    from src.pipeline.data.build.convert_to_unified import run as convert_run
    from src.pipeline.data.build.filter_training import run as filter_run

    total = 3 if args.rebuild else 2
    print(f"\n=== Step {total - 1}/{total}: Convert sources to unified JSONL ===")
    convert_args = SimpleNamespace(
        registry=args.registry,
        output=args.unified_out,
        averitec=args.averitec,
        ai2thor=args.ai2thor,
        synthetic=args.synthetic,
        intermediate_dir=args.intermediate_dir,
    )
    rc = convert_run(convert_args)
    if rc != 0:
        sys.exit(rc)

    print(f"\n=== Step {total}/{total}: Filter to training records ===")
    filter_args = SimpleNamespace(
        input=args.unified_out,
        output=args.training_out,
        verbose=args.verbose,
    )
    sys.exit(filter_run(filter_args))


if __name__ == "__main__":
    main()
