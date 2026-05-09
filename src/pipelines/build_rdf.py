from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.pipelines.result import PipelineRunResult
from src.core.registry.entity import EntityRegistry
from src.core.registry.relation import RelationRegistry

from src.adapters.ai2thor.config_loader import load_ai2thor_config
from src.adapters.ai2thor.data_source import AI2THORDataSource

from src.adapters.ai2thor.registry.entities import create_entity_registry
from src.adapters.ai2thor.registry.relations import create_relation_registry
from src.adapters.ai2thor.ontology import (
    create_ontology,
)  # your factory returning BaseOntology

from src.infra.rdf.builder import RDFGraphBuilder
from src.infra.rdf.namespaces import NamespaceConfig, NamespaceManager


def build_ai2thor_rdf(
    *,
    config_path: str,
    out_path: str,
    base_iri: Optional[str] = None,
    fmt: str = "turtle",
    strict: bool = False,
) -> PipelineRunResult:
    # 1) config + datasource``
    cfg = load_ai2thor_config(config_path)
    ds = AI2THORDataSource(cfg)

    # 2) registries + ontology
    ent_reg = EntityRegistry()
    create_entity_registry(ent_reg)

    rel_reg = RelationRegistry()
    create_relation_registry(rel_reg)

    ontology = create_ontology(entity_registry=ent_reg, relation_registry=rel_reg)

    # 3) namespaces
    ns_cfg = NamespaceConfig(base_iri=base_iri) if base_iri else NamespaceConfig()
    ns = NamespaceManager(ns_cfg)

    # 4) build graph
    builder = RDFGraphBuilder(
        ontology=ontology, namespaces=ns, strict_validation=strict
    )
    result = builder.build_from_source(ds)

    # 5) save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.graph.serialize(format=fmt), encoding="utf-8")

    # 6) cleanup
    ds.cleanup()

    return PipelineRunResult(
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
