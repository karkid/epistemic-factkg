"""
Fresh, clean knowledge graph builder.
"""

from typing import Dict, Any, Optional, Set
from rdflib import Graph, URIRef, Literal, RDF
from dataclasses import dataclass

from ..core.namespaces import create_entity_uri, create_relation_uri, create_scene_uri
from ..sources.base import DataSource, SceneData, ObjectMetadata, Relationship  
from ..ontology.base import BaseOntology, PredicateMapping

@dataclass
class BuildResult:
    """Result of building a knowledge graph."""
    graph: Graph
    num_objects: int
    num_relations: int
    scenes_processed: Set[str]


class KnowledgeGraphBuilder:
    """
    Clean, flexible knowledge graph builder.
    
    Philosophy:
    - Accept any data source via clean interface
    - Use flexible ontology for property mapping
    - Keep core logic simple and focused
    - No domain-specific assumptions
    """
    
    def __init__(self, ontology: BaseOntology):
        self.ontology = ontology
        self.graph = Graph()
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
    
    def build_from_source(self, data_source: DataSource) -> BuildResult:
        """Build knowledge graph from any data source."""
        self.graph = Graph()  # Reset
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
        
        # Process each scene
        for scene_data in data_source.get_scenes():
            self._add_scene(scene_data)
            
        return BuildResult(
            graph=self.graph,
            num_objects=self.stats["objects"],
            num_relations=self.stats["relations"], 
            scenes_processed=self.stats["scenes"].copy()
        )
    
    def build_from_scene(self, scene_data: SceneData) -> BuildResult:
        """Build knowledge graph from a single scene."""
        self.graph = Graph()
        self.stats = {"objects": 0, "relations": 0, "scenes": set()}
        
        self._add_scene(scene_data)
        
        return BuildResult(
            graph=self.graph,
            num_objects=self.stats["objects"],
            num_relations=self.stats["relations"],
            scenes_processed=self.stats["scenes"].copy()
        )
    
    def _add_scene(self, scene_data: SceneData):
        """Add a scene and all its objects to the graph."""
        scene_uri = create_scene_uri(scene_data.scene_id)
        
        # Add scene metadata if present
        if scene_data.metadata:
            for key, value in scene_data.metadata.items():
                mapping = self.ontology.get_predicate_mapping(key)
                if mapping:
                    predicate = create_relation_uri(mapping.predicate_uri)
                    literal_value = self._transform_value(value, mapping)
                    self.graph.add((scene_uri, predicate, literal_value))
        
        # Add all objects in the scene
        for obj in scene_data.objects:
            self._add_object(obj, scene_uri)
            
        # Add relationships between objects
        for relationship in scene_data.relationships:
            self._add_relationship(relationship)
        
        self.stats["scenes"].add(scene_data.scene_id)
    
    def _add_object(self, obj: ObjectMetadata, scene_uri: URIRef):
        """Add an object to the graph."""
        obj_uri = create_entity_uri(obj.object_id)
        
        # Object type
        rdf_class = self.ontology.get_object_class(obj.object_type)
        if rdf_class:
            self.graph.add((obj_uri, RDF.type, create_entity_uri(rdf_class)))
        
        # Object belongs to scene (using ontology system)
        has_object_mapping = self.ontology.get_predicate_mapping("hasObject")
        in_scene_mapping = self.ontology.get_predicate_mapping("inScene")
        
        if has_object_mapping:
            self.graph.add((scene_uri, create_relation_uri(has_object_mapping.predicate_uri), obj_uri))
            self.stats["relations"] += 1
            
        if in_scene_mapping:
            self.graph.add((obj_uri, create_relation_uri(in_scene_mapping.predicate_uri), scene_uri))
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
                predicate = create_relation_uri(mapping.predicate_uri)
                transformed_value = self._transform_value(prop_value, mapping)
                self.graph.add((obj_uri, predicate, transformed_value))
                self.stats["relations"] += 1
        
        self.stats["objects"] += 1
    
    def _add_relationship(self, relationship: Relationship):
        """Add relationship between two objects."""
        subject_uri = create_entity_uri(relationship.subject_id)
        object_uri = create_entity_uri(relationship.object_id)
        
        # Get predicate mapping from ontology
        mapping = self.ontology.get_predicate_mapping(relationship.predicate)
        if mapping:
            predicate = create_relation_uri(mapping.predicate_uri)
            self.graph.add((subject_uri, predicate, object_uri))
            self.stats["relations"] += 1
    
    def _add_position(self, obj_uri: URIRef, position: tuple[float, float, float]):
        """Add position coordinates."""
        position_mapping = self.ontology.get_predicate_mapping("position")
        if position_mapping:
            x, y, z = position
            self.graph.add((obj_uri, create_relation_uri(position_mapping.predicate_uri), Literal(f"{x},{y},{z}")))
            self.stats["relations"] += 1
    
    def _add_rotation(self, obj_uri: URIRef, rotation: tuple[float, float, float]):
        """Add rotation coordinates."""
        rotation_mapping = self.ontology.get_predicate_mapping("rotation")
        if rotation_mapping:
            x, y, z = rotation
            self.graph.add((obj_uri, create_relation_uri(rotation_mapping.predicate_uri), Literal(f"{x},{y},{z}")))
            self.stats["relations"] += 1
    
    def _transform_value(self, value: Any, mapping: PredicateMapping) -> Literal:
        """Transform a value using the mapping's transform function."""
        if mapping.transform_func:
            value = mapping.transform_func(value)
        return Literal(value)
    
    def export_graph(self, format: str = "turtle") -> str:
        """Export graph in specified format."""
        return self.graph.serialize(format=format)
