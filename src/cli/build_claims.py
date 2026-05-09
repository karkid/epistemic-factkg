#!/usr/bin/env python3
"""Build semantic claims from a knowledge graph TTL file."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.adapters.ai2thor.semantics.semantic_rules import get_preferred_receptacles
from src.adapters.ai2thor.ids.object_types import ObjectType
from src.core.claims.labels import Pramana
from src.core.claims.claim_generator import ClaimGenerator
from src.core.claims.types import ClaimCorpus
from src.adapters.ai2thor.semantics.entity_lexicon import (
    create_ai2thor_object_type_lexicon,
)
from src.adapters.ai2thor.semantics.predicate_lexicon import create_predicate_lexicon
from src.adapters.ai2thor.nlg.template import Ai2ThorTemplate
from src.core.nlg.triple_realizer import TripleRealizer
from src.core.graph.types import TripleList
from src.infra.rdf.io.ttl import load_triples_from_ttl
from src.infra.rdf.query.engine import TripleQueryEngine
from src.infra.rdf.formatter import ai2thor_object_type_from_entity_id
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildClaimsResult:
    ttl_path: str
    total_triples: int
    contexts: List[str]
    context_triples: Dict[str, TripleList]
    claim_corpus: ClaimCorpus
    output_files: List[str]


def build_claims(
    ttl_path: str,
    output_dir: str = "out",
    max_contexts: Optional[int] = None,
    n_claims: int = 500,
    add_corruption: bool = True,
    verbose: bool = False,
) -> BuildClaimsResult:
    if verbose:
        print(f"Loading triples from: {ttl_path}")

    triples = load_triples_from_ttl(ttl_path)
    query_engine = TripleQueryEngine(triples)

    if verbose:
        print(f"Loaded {len(triples)} triples")

    group_by_context = query_engine.group_by_namespace(namespace="contexts")
    contexts = sorted(group_by_context.keys())

    if max_contexts and max_contexts > 0:
        contexts = contexts[:max_contexts]

    if verbose:
        print(f"Processing {len(contexts)} contexts: {contexts}")

    pred_lex = create_predicate_lexicon()
    ent_lex = create_ai2thor_object_type_lexicon()
    realizer = TripleRealizer(
        template=Ai2ThorTemplate(predicate_lexicon=pred_lex),
        pred_lexicon=pred_lex,
        ent_lexicon=ent_lex,
        normallizer=ai2thor_object_type_from_entity_id,
    )

    total_corpus = ClaimCorpus()

    for context in contexts:
        if verbose:
            print(f"Processing context: {context}")

        context_triples = group_by_context[context]
        claim_generator = ClaimGenerator(
            realizer=realizer,
            context_id=context,
            triples=context_triples,
            context_type="floorplan",
            generator="ai2thor-scene-simulator",
            source="sensor",
            source_type=Pramana.PERCEPTION,
            receptacle_mapper=get_preferred_receptacles,
        )

        claim_generator.generate_one_hop(
            n_claims=n_claims, add_corruption=add_corruption
        )
        claim_generator.generate_conjunction(
            n_claims=n_claims, add_corruption=add_corruption
        )
        claim_generator.generate_negation(
            n_claims=n_claims, add_corruption=add_corruption
        )
        claim_generator.generate_absence(
            n_claims=n_claims,
            add_corruption=add_corruption,
            object_universe=[ot.value for ot in ObjectType],
        )

        total_corpus.claims.extend(claim_generator.corpus.claims)

        if verbose:
            print(
                f"Generated {len(claim_generator.corpus.claims)} claims for {context}"
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(out_dir / "claims_all.jsonl")
    total_corpus.save_to_jsonl(output_file)

    if verbose:
        print(f"Total claims: {len(total_corpus.claims)}")
        print(f"Saved to: {output_file}")

    return BuildClaimsResult(
        ttl_path=ttl_path,
        total_triples=len(triples),
        contexts=contexts,
        context_triples=group_by_context,
        claim_corpus=total_corpus,
        output_files=[output_file],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build semantic claims from a knowledge graph TTL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ttl_path", help="Path to the TTL file")
    parser.add_argument("--output-dir", "-o", default="out", help="Output directory")
    parser.add_argument(
        "--max-contexts", type=int, default=None, help="Max contexts to process"
    )
    parser.add_argument(
        "--n-claims", type=int, default=500, help="Claims per generation method"
    )
    parser.add_argument(
        "--no-corruption", action="store_true", help="Disable corrupted claims"
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    ttl_path = Path(args.ttl_path)
    if not ttl_path.exists():
        print(f"Error: TTL file not found: {ttl_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = build_claims(
            ttl_path=str(ttl_path),
            output_dir=args.output_dir,
            max_contexts=args.max_contexts,
            n_claims=args.n_claims,
            add_corruption=not args.no_corruption,
            verbose=args.verbose,
        )

        print("=" * 60)
        print("Build Claims: SUCCESS")
        print(f"Input TTL    : {result.ttl_path}")
        print(f"Triples      : {result.total_triples}")
        print(f"Contexts     : {len(result.contexts)}")
        print(f"Total claims : {len(result.claim_corpus.claims)}")
        for f in result.output_files:
            print(f"  -> {f}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
