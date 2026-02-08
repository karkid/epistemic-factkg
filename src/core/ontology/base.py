from __future__ import annotations

from typing import Dict, Set, Callable, Optional

from src.core.registry.entity import EntityInfo
from src.core.registry.relation import RelationInfo, RelationType
from src.core.ontology.mappings import PredicateMapping, EntityMapping


class BaseOntology:
    """Base ontology for KG construction and NLG."""

    def __init__(self):

        # source -> mapping
        self._by_source: Dict[str, PredicateMapping] = {}

        # uri -> mapping
        self._by_uri: Dict[str, PredicateMapping] = {}

        # object types
        self._object_types: Dict[str, EntityMapping] = {}

    def register_predicate(
        self,
        *,
        info: RelationInfo,
        uri: str,
        transform: Optional[Callable] = None,
    ) -> None:

        mapping = PredicateMapping(
            info=info,
            uri=uri,
            transform=transform,
        )

        self._by_source[info.name] = mapping
        self._by_uri[uri] = mapping


    def register_object_type(
        self,
        info: EntityInfo,
        uri: str, # uri
        transform: Optional[Callable] = None,
    ) -> None:

        self._object_types[info.type] = EntityMapping(
            info=info,
            uri=uri,
            transform=transform,
        )

    def by_source(self, field: str) -> Optional[PredicateMapping]:

        return self._by_source.get(field)

    def by_uri(self, uri: str) -> Optional[PredicateMapping]:

        return self._by_uri.get(uri)

    def predicates_by_type(
        self,
        relation_type: RelationType,
    ) -> Set[str]:

        return {
            m.uri
            for m in self._by_uri.values()
            if m.info.type == relation_type
        }

    def is_predicate_type(
        self,
        uri: str,
        relation_type: RelationType,
    ) -> bool:

        m = self._by_uri.get(uri)

        return m is not None and m.info.type == relation_type


    def is_state(self, uri: str) -> bool:
        return self.is_predicate_type(uri, RelationType.STATE)

    def is_spatial(self, uri: str) -> bool:
        return self.is_predicate_type(uri, RelationType.SPATIAL)

    def is_data(self, uri: str) -> bool:
        return self.is_predicate_type(uri, RelationType.DATA)

    def is_object_relation(self, uri: str) -> bool:
        return self.is_predicate_type(uri, RelationType.OBJECT)

    def has_object_type(self, t: str) -> bool:

        return t in self._object_types

    def object_class(self, t: str) -> Optional[EntityMapping]:

        return self._object_types.get(t)

    def all_object_types(self) -> Set[str]:

        return set(self._object_types)

    def all_predicates(self) -> Set[str]:

        return set(self._by_uri)
