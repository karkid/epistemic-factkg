# src/infra/rdf/builder.py
from __future__ import annotations

from typing import Any, Optional

from rdflib import Graph, URIRef, Literal, RDF

from src.infra.rdf.namespaces import NamespaceManager, NamespaceConfig
from src.core.graph.types import Graph as SceneGraph, Object, Relationship
from src.core.ports.graph.graph_data_source import (
    GraphDataSource,
)  # your DataSource port
from src.core.ontology import BaseOntology
from src.utils.exceptions import BuildError, ValidationError, DataSourceError
from src.infra.rdf.result import GraphBuildResult

# ---------- Builder ----------


class RDFGraphBuilder:
    """
    Generic RDF graph builder.

    Inputs:
      - SceneGraph (objects + relationships + metadata)
      - BaseOntology mapping source keys -> predicate URIs
      - NamespaceManager for entity/scene URIs

    Notes:
      - Ontology predicate/object-type mappings are assumed to already store full URIs.
      - NamespaceManager is used to mint *entity* and *scene* node URIs.
    """

    # defaults for "core" relations used by builder; keep stable across domains
    HAS_OBJECT_KEY = "hasObject"
    IN_SCENE_KEY = "inScene"
    POSITION_KEY = "position"
    ROTATION_KEY = "rotation"

    def __init__(
        self,
        *,
        ontology: BaseOntology,
        namespaces: NamespaceManager | None = None,
        strict_validation: bool = False,
    ) -> None:
        self.ontology = ontology
        self.namespaces = namespaces or NamespaceManager(NamespaceConfig())
        self.strict_validation = strict_validation

        self.graph: Graph = Graph()
        self._stats = {"objects": 0, "relations": 0, "contexts": set()}
        self._result: Optional[GraphBuildResult] = None

    # ---------------- Public API ----------------

    def build_from_source(self, data_source: GraphDataSource) -> GraphBuildResult:
        if data_source is None:
            raise BuildError("data_source cannot be None")

        self._reset()

        try:
            available = data_source.get_available_graph_ids()
            if not available:
                self._result.add_warning("No scenes available from data source")
                return self._finalize()

            processed = 0
            for scene in data_source.get_graphs():
                try:
                    self._add_scene(scene)
                    processed += 1
                except Exception as e:
                    msg = f"Failed to process scene {getattr(scene, 'graph_id', 'unknown')}: {e}"
                    self._result.add_error(msg)
                    if self.strict_validation:
                        raise BuildError(msg) from e

            if processed == 0:
                self._result.add_warning("No scenes were successfully processed")

        except Exception as e:
            if isinstance(e, BuildError):
                raise
            raise DataSourceError(f"Data source failed during processing: {e}") from e

        return self._finalize()

    def build_from_scene(self, scene: SceneGraph) -> GraphBuildResult:
        if scene is None:
            raise ValidationError("scene cannot be None")
        if not getattr(scene, "graph_id", None):
            raise ValidationError("scene.graph_id is required")

        self._reset()

        try:
            self._add_scene(scene)
        except Exception as e:
            msg = f"Failed to process scene {scene.graph_id}: {e}"
            self._result.add_error(msg)
            if self.strict_validation:
                raise BuildError(msg) from e

        return self._finalize()

    def export_graph(self, format: str = "turtle") -> str:
        return self.graph.serialize(format=format)

    # ---------------- Internals ----------------

    def _reset(self) -> None:
        self.graph = Graph()
        self._stats = {"objects": 0, "relations": 0, "contexts": set()}
        self._result = GraphBuildResult(
            graph=self.graph,
            num_objects=0,
            num_relations=0,
            contexts_processed=set(),
        )

    def _finalize(self) -> GraphBuildResult:
        assert self._result is not None
        self._result.num_objects = self._stats["objects"]
        self._result.num_relations = self._stats["relations"]
        self._result.contexts_processed = set(self._stats["contexts"])
        return self._result

    def _add_scene(self, scene: SceneGraph) -> None:
        if not scene.graph_id:
            raise ValidationError("SceneGraph.graph_id is required")

        scene_uri = self.namespaces.context_uri(scene.graph_id)

        # Scene metadata (optional)
        if scene.metadata:
            for k, v in scene.metadata.items():
                self._try_add_data_literal(scene_uri, k, v, where=f"scene metadata {k}")

        # Objects
        for obj in scene.objects:
            try:
                self._add_object(obj, scene_uri)
            except Exception as e:
                msg = (
                    f"Failed to add object {getattr(obj, 'object_id', 'unknown')}: {e}"
                )
                self._result.add_warning(msg)
                if self.strict_validation:
                    raise ValidationError(msg) from e

        # Relationships
        for rel in scene.relationships or []:
            try:
                self._add_relationship(rel)
            except Exception as e:
                msg = f"Failed to add relationship {rel.subject_id} -[{rel.predicate}]-> {rel.object_id}: {e}"
                self._result.add_warning(msg)
                if self.strict_validation:
                    raise ValidationError(msg) from e

        self._stats["contexts"].add(scene.graph_id)

    def _add_object(self, obj: Object, scene_uri: URIRef) -> None:
        if not obj.object_id:
            raise ValidationError("Object.object_id is required")

        obj_uri = self.namespaces.entity_uri(obj.object_id)

        # rdf:type for object class
        obj_class = self.ontology.object_class(obj.object_type)
        if obj_class is not None:
            self.graph.add((obj_uri, RDF.type, URIRef(obj_class.uri)))

        # scene membership (two-way)
        self._try_add_relation(scene_uri, self.HAS_OBJECT_KEY, obj_uri)
        self._try_add_relation(obj_uri, self.IN_SCENE_KEY, scene_uri)

        # position/rotation [optional]
        if obj.position is not None:
            self._try_add_data_literal(
                obj_uri,
                self.POSITION_KEY,
                f"{obj.position[0]},{obj.position[1]},{obj.position[2]}",
                where="position",
            )
        if obj.rotation is not None:
            self._try_add_data_literal(
                obj_uri,
                self.ROTATION_KEY,
                f"{obj.rotation[0]},{obj.rotation[1]},{obj.rotation[2]}",
                where="rotation",
            )

        # properties -> literals
        for prop_name, prop_value in (obj.properties or {}).items():
            self._try_add_data_literal(
                obj_uri, prop_name, prop_value, where=f"property {prop_name}"
            )

        self._stats["objects"] += 1

    def _add_relationship(self, rel: Relationship) -> None:
        subj = self.namespaces.entity_uri(rel.subject_id)
        obj = self.namespaces.entity_uri(rel.object_id)
        self._try_add_relation(subj, rel.predicate, obj)

    # ------------- Add helpers using ontology mappings -------------

    def _try_add_relation(self, s: URIRef, predicate_key: str, o: URIRef) -> None:
        mapping = self.ontology.by_source(predicate_key)
        if mapping is None:
            # not mapped -> skip quietly
            return

        try:
            p = URIRef(mapping.uri)
            self.graph.add((s, p, o))
            self._stats["relations"] += 1
        except Exception as e:
            msg = f"Failed to add relation {predicate_key}: {e}"
            self._result.add_warning(msg)
            if self.strict_validation:
                raise ValidationError(msg) from e

    def _try_add_data_literal(
        self, s: URIRef, predicate_key: str, value: Any, *, where: str
    ) -> None:
        mapping = self.ontology.by_source(predicate_key)
        if mapping is None:
            return
        try:
            v = value
            if getattr(mapping, "transform", None):
                v = mapping.transform(v)
            p = URIRef(mapping.uri)
            self.graph.add((s, p, Literal(v)))
            self._stats["relations"] += 1
        except Exception as e:
            msg = f"Failed to add {where} via predicate {predicate_key}: {e}"
            self._result.add_warning(msg)
            if self.strict_validation:
                raise ValidationError(msg) from e
