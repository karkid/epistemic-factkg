from abc import ABC

from src.core.registry.relation import RelationType

class KGInclusionPolicy(ABC):

    def should_include_relationship_type(self, relationship: RelationType) -> bool:
        """Determine if a given relationship should be included in the KG."""
        return True

    def should_include_entity_type(self, entity_type: str) -> bool:
        """Determine if a given entity type should be included in the KG."""
        return True

    def should_include_property(self, property_name: str) -> bool:
        """Determine if a given property should be included in the KG."""
        return True
    
    def should_include_scene(self, scene_id: str) -> bool:
        """Determine if a given scene should be included in the KG."""
        return True

    def should_include_relationship(self, predicate: str) -> bool:
        """Determine if a specific relationship instance should be included in the KG."""
        return True

    def should_include_entity(self, entity_id: str) -> bool:
        """Determine if a specific entity instance should be included in the KG."""
        return True
