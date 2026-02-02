"""
Example showing how to use the fresh knowledge graph library.
"""

from knowledge_graph import (
    DataSource, SceneData, ObjectMetadata, Relationship,
    BaseOntology, PredicateMapping, RelationType, 
    KnowledgeGraphBuilder
)


# 1. Create a simple data source
class SimpleDataSource(DataSource):
    """Example data source with hardcoded data."""
    
    def get_scenes(self):
        # Example scene with some objects
        objects = [
            ObjectMetadata(
                object_id="mug_001",
                object_type="Mug", 
                properties={"color": "red", "material": "ceramic", "fillLevel": 0.8},
                position=(1.0, 0.5, 2.0)
            ),
            ObjectMetadata(
                object_id="table_001",
                object_type="Table",
                properties={"color": "brown", "material": "wood"},
                position=(1.0, 0.0, 2.0)
            )
        ]
        
        # Relationships between objects
        relationships = [
            Relationship("mug_001", "isOn", "table_001"),         # Mug is on table
            Relationship("mug_001", "nextTo", "table_001"),       # Mug is next to table  
        ]
        
        scene = SceneData(
            scene_id="kitchen_01", 
            objects=objects,
            relationships=relationships,    # Add relationships!
            metadata={"room_type": "kitchen"}
        )
        
        yield scene
    
    def get_scene_by_id(self, scene_id: str):
        # Simple implementation
        for scene in self.get_scenes():
            if scene.scene_id == scene_id:
                return scene
        raise ValueError(f"Scene {scene_id} not found")


def main():
    """Example usage."""
    
    # 2. Configure ontology with predicate mappings
    ontology = BaseOntology()  # Core structural relations auto-registered!
    
    # Register domain-specific object properties (data properties)
    ontology.register_predicate("color", "hasColor")
    ontology.register_predicate("material", "hasMaterial") 
    ontology.register_predicate("fillLevel", "hasFillLevel")
    ontology.register_predicate("room_type", "roomType")
    
    # Register domain-specific relationships (object relations)  
    ontology.register_predicate("isOn", "isOn", RelationType.OBJECT_RELATION)
    ontology.register_predicate("nextTo", "nextTo", RelationType.SPATIAL)
    
    # Register object types
    ontology.register_object_type("Mug", "Mug")
    ontology.register_object_type("Table", "Table")
    
    # 3. Create builder and build graph
    builder = KnowledgeGraphBuilder(ontology)
    data_source = SimpleDataSource()
    
    result = builder.build_from_source(data_source)
    
    # 4. Export results
    print(f"Built graph with {result.num_objects} objects and {result.num_relations} relations")
    print(f"Processed scenes: {result.scenes_processed}")
    
    # Export to turtle format
    turtle_output = result.graph.serialize(format="turtle")
    print("\\nGenerated RDF (Turtle format):")
    print(turtle_output)


if __name__ == "__main__":
    main()