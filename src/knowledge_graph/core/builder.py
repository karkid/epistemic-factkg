"""
Fresh, clean knowledge graph builder with enhanced error handling and flexibility.
"""

from typing import Any, Optional, Set, List
from rdflib import Graph, URIRef, Literal
from dataclasses import dataclass

from ..core.namespaces import (
    NamespaceManager,
    DEFAULT_NAMESPACE_MANAGER,
)
from knowledge_graph.sources.base import DataSource, SceneData, ObjectMetadata, Relationship
from knowledge_graph.ontology.base import BaseOntology, CorePredicates, PredicateMapping
from utils.exceptions import BuildError, ValidationError, DataSourceError


@dataclass
class BuildResult:
    """Result of building a knowledge graph with comprehensive statistics."""

    graph: Graph
    num_objects: int
    num_relations: int
    scenes_processed: Set[str]
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        """Initialize warning and error lists if not provided."""
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        """Return True if build completed without errors."""
        return len(self.errors) == 0

    @property
    def total_triples(self) -> int:
        """Return total number of triples in the graph."""
        return len(self.graph)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)


class KnowledgeGraphBuilder:
    """
    Clean, flexible knowledge graph builder with enhanced error handling.

    Philosophy:
    - Accept any data source via clean interface
    - Use flexible ontology for property mapping
    - Keep core logic simple and focused
    - No domain-specific assumptions
    - Comprehensive error handling and reporting
    - Configurable namespace management

    Args:
        ontology: The ontology defining property mappings
        namespace_manager: Optional namespace manager for custom URI patterns
        strict_validation: If True, raises exceptions on validation errors
    """

    def __init__(
        self,
        ontology: BaseOntology,
        namespace_manager: Optional[NamespaceManager] = None,
        strict_validation: bool = False,
    ):
        """Initialize the knowledge graph builder."""
        self.ontology = ontology
        self.namespace_manager = namespace_manager or DEFAULT_NAMESPACE_MANAGER
        self.strict_validation = strict_validation
        self.graph = Graph()
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
        self._current_result: Optional[BuildResult] = None

    def build_from_source(self, data_source: DataSource) -> BuildResult:
        """Build knowledge graph from any data source with comprehensive error handling.

        Args:
            data_source: The data source to build from

        Returns:
            BuildResult containing the graph and statistics

        Raises:
            BuildError: If critical errors occur during building
            DataSourceError: If data source fails to provide data
        """
        if data_source is None:
            raise BuildError("Data source cannot be None")

        # Reset state
        self.graph = Graph()
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
        self._current_result = BuildResult(
            graph=self.graph, num_objects=0, num_relations=0, scenes_processed=set()
        )

        try:
            # Get available scenes first to validate data source
            available_scenes = data_source.get_available_scenes()
            if not available_scenes:
                self._current_result.add_warning("No scenes available from data source")
                return self._current_result

            # Process each scene
            scenes_processed = 0
            for scene_data in data_source.get_scenes():
                try:
                    self._add_scene(scene_data)
                    scenes_processed += 1
                except Exception as e:
                    error_msg = f"Failed to process scene {getattr(scene_data, 'scene_id', 'unknown')}: {e}"
                    self._current_result.add_error(error_msg)
                    if self.strict_validation:
                        raise BuildError(error_msg) from e

            if scenes_processed == 0:
                self._current_result.add_warning(
                    "No scenes were successfully processed"
                )

        except Exception as e:
            if isinstance(e, BuildError):
                raise
            raise DataSourceError(f"Data source failed during processing: {e}") from e

        # Finalize result
        self._current_result.num_objects = self.stats["objects"]
        self._current_result.num_relations = self.stats["relations"]
        self._current_result.scenes_processed = self.stats["scenes"].copy()

        return self._current_result

    def build_from_scene(self, scene_data: SceneData) -> BuildResult:
        """Build knowledge graph from a single scene with validation.

        Args:
            scene_data: The scene data to build from

        Returns:
            BuildResult containing the graph and statistics

        Raises:
            BuildError: If critical errors occur during building
            ValidationError: If scene data is invalid
        """
        if scene_data is None:
            raise ValidationError("Scene data cannot be None")

        if not scene_data.scene_id:
            raise ValidationError("Scene data must have a valid scene_id")

        # Reset state
        self.graph = Graph()
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
        self._current_result = BuildResult(
            graph=self.graph, num_objects=0, num_relations=0, scenes_processed=set()
        )

        try:
            self._add_scene(scene_data)
        except Exception as e:
            error_msg = f"Failed to process scene {scene_data.scene_id}: {e}"
            self._current_result.add_error(error_msg)
            if self.strict_validation:
                raise BuildError(error_msg) from e

        # Finalize result
        self._current_result.num_objects = self.stats["objects"]
        self._current_result.num_relations = self.stats["relations"]
        self._current_result.scenes_processed = self.stats["scenes"].copy()

        return self._current_result

    def _add_scene(self, scene_data: SceneData):
        """Add a scene and all its objects to the graph with error handling."""
        try:
            scene_uri = self.namespace_manager.create_scene_uri(scene_data.scene_id)

            # Add scene metadata if present
            if scene_data.metadata:
                for key, value in scene_data.metadata.items():
                    try:
                        mapping = self.ontology.get_predicate_mapping(key)
                        if mapping:
                            predicate = self.namespace_manager.create_relation_uri(
                                mapping.predicate_uri
                            )
                            literal_value = self._transform_value(value, mapping)
                            self.graph.add((scene_uri, predicate, literal_value))
                    except Exception as e:
                        warning_msg = f"Failed to add scene metadata {key}={value}: {e}"
                        if self._current_result:
                            self._current_result.add_warning(warning_msg)
                        if self.strict_validation:
                            raise ValidationError(warning_msg) from e

            # Add all objects in the scene
            for obj in scene_data.objects:
                try:
                    self._add_object(obj, scene_uri)
                except Exception as e:
                    warning_msg = f"Failed to add object {obj.object_id}: {e}"
                    if self._current_result:
                        self._current_result.add_warning(warning_msg)
                    if self.strict_validation:
                        raise ValidationError(warning_msg) from e

            # Add relationships between objects
            for relationship in scene_data.relationships:
                try:
                    self._add_relationship(relationship)
                except Exception as e:
                    warning_msg = f"Failed to add relationship {relationship.subject_id} -> {relationship.object_id}: {e}"
                    if self._current_result:
                        self._current_result.add_warning(warning_msg)
                    if self.strict_validation:
                        raise ValidationError(warning_msg) from e

            self.stats["scenes"].add(scene_data.scene_id)

        except Exception as e:
            if isinstance(e, (ValidationError, BuildError)):
                raise
            raise BuildError(f"Failed to add scene {scene_data.scene_id}: {e}") from e

    def _add_object(self, obj: ObjectMetadata, scene_uri: URIRef):
        """Add an object to the graph."""
        obj_uri = self.namespace_manager.create_entity_uri(obj.object_id)

        # Object type
        rdf_class = self.ontology.get_object_class(obj.object_type)
        if rdf_class:
            self.graph.add(
                (
                    obj_uri,
                    self.namespace_manager.get_type_predicate(),
                    self.namespace_manager.create_entity_uri(rdf_class),
                )
            )

        # Object belongs to scene (using ontology system)
        has_object_mapping = self.ontology.get_predicate_mapping(
            CorePredicates.HAS_OBJECT.name
        )
        in_scene_mapping = self.ontology.get_predicate_mapping(
            CorePredicates.IN_SCENE.name
        )

        if has_object_mapping:
            self.graph.add(
                (
                    scene_uri,
                    self.namespace_manager.create_relation_uri(
                        has_object_mapping.predicate_uri
                    ),
                    obj_uri,
                )
            )
            self.stats["relations"] += 1

        if in_scene_mapping:
            self.graph.add(
                (
                    obj_uri,
                    self.namespace_managercreate_relation_uri(
                        in_scene_mapping.predicate_uri
                    ),
                    scene_uri,
                )
            )
            self.stats["relations"] += 1

        # Position if available
        if obj.position:
            self._add_position(obj_uri, obj.position)

        # Rotation if available
        if obj.rotation:
            self._add_rotation(obj_uri, obj.rotation)

        # All other properties
        for prop_name, prop_value in obj.properties.items():
            mapping = self.ontology.get_predicate_mapping(prop_name)
            if mapping:
                predicate = self.namespace_manager.create_relation_uri(
                    mapping.predicate_uri
                )
                transformed_value = self._transform_value(prop_value, mapping)
                self.graph.add((obj_uri, predicate, transformed_value))
                self.stats["relations"] += 1

        self.stats["objects"] += 1

    def _add_relationship(self, relationship: Relationship):
        """Add relationship between two objects using namespace manager."""
        subject_uri = self.namespace_manager.create_entity_uri(relationship.subject_id)
        object_uri = self.namespace_manager.create_entity_uri(relationship.object_id)

        # Map predicate through ontology
        mapping = self.ontology.get_predicate_mapping(relationship.predicate)
        if mapping:
            predicate_uri = self.namespace_manager.create_relation_uri(
                mapping.predicate_uri
            )
            self.graph.add((subject_uri, predicate_uri, object_uri))
            self.stats["relations"] += 1

    def _add_position(self, obj_uri: URIRef, position: tuple[float, float, float]):
        """Add position coordinates."""
        position_mapping = self.ontology.get_predicate_mapping(
            CorePredicates.POSITION.name
        )
        if position_mapping:
            x, y, z = position
            self.graph.add(
                (
                    obj_uri,
                    self.namespace_manager.create_relation_uri(
                        position_mapping.predicate_uri
                    ),
                    Literal(f"{x},{y},{z}"),
                )
            )
            self.stats["relations"] += 1

    def _add_rotation(self, obj_uri: URIRef, rotation: tuple[float, float, float]):
        """Add rotation coordinates."""
        rotation_mapping = self.ontology.get_predicate_mapping(
            CorePredicates.ROTATION.name
        )
        if rotation_mapping:
            x, y, z = rotation
            self.graph.add(
                (
                    obj_uri,
                    self.namespace_manager.create_relation_uri(
                        rotation_mapping.predicate_uri
                    ),
                    Literal(f"{x},{y},{z}"),
                )
            )
            self.stats["relations"] += 1

    def _transform_value(self, value: Any, mapping: PredicateMapping) -> Literal:
        """Transform a value using the mapping's transform function."""
        if mapping.transform_func:
            value = mapping.transform_func(value)
        return Literal(value)

    def export_graph(self, format: str = "turtle") -> str:
        """Export graph in specified format."""
        return self.graph.serialize(format=format)
