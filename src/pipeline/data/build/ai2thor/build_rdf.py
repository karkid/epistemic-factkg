from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.ontology.registry import EntityRegistry, RelationRegistry
from src.adapters.ai2thor.scene.loader import load_ai2thor_config
from src.adapters.ai2thor.scene.data_source import AI2THORDataSource
from src.adapters.ai2thor.knowledge.registry import (
    create_entity_registry,
    create_relation_registry,
)
from src.adapters.ai2thor.knowledge.ontology import create_ontology
from src.infra.rdf.builder import RDFGraphBuilder
from src.infra.rdf.namespaces import NamespaceConfig, NamespaceManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BuildRDFResult:
    success: bool
    out_path: str
    format: str
    num_objects: int
    num_relations: int
    total_triples: int
    contexts_processed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BuildRDFResultSummary:
    def __init__(self, results: List[BuildRDFResult] = None):
        self.results = results if results is not None else []

    def add_result(self, result: BuildRDFResult):
        self.results.append(result)

    def print_summary(
        self, logger_func=print, max_items: int = 3, show_lists: bool = False
    ):
        total_triples = sum(r.total_triples for r in self.results)
        logger_func("=" * 72)
        logger_func(f"Scenes processed  : {len(self.results)}")
        logger_func(f"Total objects     : {sum(r.num_objects for r in self.results)}")
        logger_func(f"Total relations   : {sum(r.num_relations for r in self.results)}")
        logger_func(f"Total triples     : {total_triples}")
        logger_func(f"Total warnings    : {sum(len(r.warnings) for r in self.results)}")
        logger_func(f"Total errors      : {sum(len(r.errors) for r in self.results)}")
        logger_func("=" * 72)

        for res in self.results:
            status = "OK" if res.success else "FAILED"
            logger_func(f"\nBuild RDF: {status}")
            logger_func(f"Output    : {res.out_path}")
            logger_func(f"Format    : {res.format}")
            logger_func(f"Triples   : {res.total_triples}")
            logger_func(f"Warnings  : {len(res.warnings)}")
            logger_func(f"Errors    : {len(res.errors)}")
            if res.warnings:
                for w in res.warnings[:max_items]:
                    logger_func(f"  warn: {w}")
            if res.errors:
                for e in res.errors[:max_items]:
                    logger_func(f"  err:  {e}")


def build_ai2thor_rdf(
    *,
    config_path: str,
    out_path: str,
    base_iri: Optional[str] = None,
    fmt: str = "turtle",
    strict: bool = False,
) -> BuildRDFResult:
    cfg = load_ai2thor_config(config_path)
    ds = AI2THORDataSource(cfg)

    ent_reg = EntityRegistry()
    create_entity_registry(ent_reg)

    rel_reg = RelationRegistry()
    create_relation_registry(rel_reg)

    ontology = create_ontology(entity_registry=ent_reg, relation_registry=rel_reg)

    ns_cfg = NamespaceConfig(base_iri=base_iri) if base_iri else NamespaceConfig()
    ns = NamespaceManager(ns_cfg)

    builder = RDFGraphBuilder(
        ontology=ontology, namespaces=ns, strict_validation=strict
    )
    result = builder.build_from_source(ds)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.graph.serialize(format=fmt), encoding="utf-8")

    ds.cleanup()

    return BuildRDFResult(
        success=result.success,
        out_path=out_path,
        format=fmt,
        num_objects=result.num_objects,
        num_relations=result.num_relations,
        total_triples=result.total_triples,
        contexts_processed=sorted(result.contexts_processed),
        warnings=result.warnings,
        errors=result.errors,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RDF (Turtle) from AI2-THOR scenes.")
    ap.add_argument("--config", required=True, help="Path to ai2thor YAML config")
    ap.add_argument("--out", required=True, help="Output .ttl path")
    ap.add_argument("--base-iri", default=None, help="Override base IRI for namespaces")
    ap.add_argument("--format", default="turtle", help="rdflib serialize format")
    ap.add_argument("--strict", action="store_true", help="Fail on validation errors")
    ap.add_argument("--json", action="store_true", help="Print result as JSON")
    ap.add_argument("--verbose", action="store_true", help="Print full warnings/errors")

    args = ap.parse_args()

    res: BuildRDFResult = build_ai2thor_rdf(
        config_path=args.config,
        out_path=args.out,
        base_iri=args.base_iri,
        fmt=args.format,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        summary = BuildRDFResultSummary()
        summary.add_result(res)
        summary.print_summary(show_lists=args.verbose)


if __name__ == "__main__":
    main()
