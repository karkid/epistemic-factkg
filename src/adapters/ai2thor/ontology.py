from __future__ import annotations

from src.core.ontology import BaseOntology
from src.core.registry import EntityRegistry, RelationRegistry
from src.infra.rdf.namespaces import NamespaceConfig, NamespaceManager


def create_ontology(
    *,
    relation_registry: RelationRegistry,
    entity_registry: EntityRegistry,
    base_iri: str | None = None,
) -> BaseOntology:
    ontology = BaseOntology()

    ns = NamespaceManager(
        NamespaceConfig(base_iri=base_iri or "http://epistemicfactkg.org/")
    )

    # relation_registry: rid -> RelationInfo
    for rid, info in relation_registry.get_all().items():
        ontology.register_predicate(info=info, uri=str(ns.relation_uri(rid)))

    # entity_registry: eid -> EntityInfo
    for eid, info in entity_registry.get_all().items():
        ontology.register_object_type(info=info, uri=str(ns.entity_uri(eid)))

    return ontology
