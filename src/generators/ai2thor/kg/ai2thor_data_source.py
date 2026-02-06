"""
AI2-THOR specific data source implementation.

Integrates with base knowledge_graph architecture.
"""

from typing import Iterator, Dict, List, Any
from pathlib import Path
import yaml
from ai2thor.controller import Controller
from knowledge_graph.ontology.base import CorePredicates
from knowledge_graph.sources.base import (
    DataSource,
    SceneData,
    ObjectMetadata,
    Relationship,
)
from utils.dot_namespace import to_namespace
from .object_type import (
    AI2THOR_CONTAINER_OBJECT_TYPES,
    AI2THOR_SURFACE_OBJECT_TYPES,
    AI2THOR_HANGING_OBJECT_TYPES,
)
from .relation_type import (
    AI2THOR_ATTRIBUTE_STATE_MAPPING,
    AI2THOR_ATTRIBUTE_TYPES,
    AI2THOR_LITERAL_RELATION_MAPPING,
    AI2THOR_SPATIAL_RELATION_TYPES,
)
from utils.exceptions import (
    ConfigurationError,
)


class AI2THORDataSource(DataSource):
    """
    AI2-THOR data source that creates its own controller from config.

    Self-contained - reads config and manages AI2-THOR controller internally.
    """

    # Cache namespace conversions for better performance
    _cached_container_types = None
    _cached_surface_types = None
    _cached_hanging_types = None
    _cached_attribute_state_types = None
    _cached_value_relation_custom_mapping = None
    _cached_spatial_relation_types = None

    def __init__(self, config_path: str = None):
        """
        Initialize AI2-THOR data source with config.

        Args:
            config_path: Path to thor.yaml config file.
                        If None, uses default location.
        """
        if config_path is None:
            # Default to project configs folder
            config_path = (
                Path(__file__).parent.parent.parent.parent.parent
                / "configs"
                / "ai2thor_default.yaml"
            )

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.kg_policy = self.config.get("knowledge_graph_policy", {})

        # Initialize cached namespace conversions
        self._init_cached_types()

        # Initialize controller after caching is set up
        self.controller = self._create_controller()

    def _load_config(self) -> Dict[str, Any]:
        """Load AI2-THOR configuration from YAML with error handling."""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Invalid configuration format in {self.config_path}: expected dictionary"
                )

            # Validate required sections
            if "scenes" not in config:
                raise ConfigurationError("Configuration must include 'scenes' section")

            if not config["scenes"]:
                raise ConfigurationError(
                    "Configuration must specify at least one scene"
                )

            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration {self.config_path}: {e}"
            ) from e
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Unexpected error loading configuration: {e}"
            ) from e

    def _create_controller(self) -> Controller:
        """Create AI2-THOR controller from enhanced config settings."""
        controller_settings = self.config.get("controller", {})

        # Filter out None values and comments
        filtered_settings = {
            k: v
            for k, v in controller_settings.items()
            if v is not None and not k.startswith("#")
        }

        return Controller(
            **filtered_settings,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            renderSemanticSegmentation=False,
            renderImage=False,
            visibilityDistance=1.5,
        )

    def _init_cached_types(self) -> None:
        """Initialize cached namespace conversions for better performance."""
        if AI2THORDataSource._cached_container_types is None:
            AI2THORDataSource._cached_container_types = to_namespace(
                AI2THOR_CONTAINER_OBJECT_TYPES
            )
            AI2THORDataSource._cached_surface_types = to_namespace(
                AI2THOR_SURFACE_OBJECT_TYPES
            )
            AI2THORDataSource._cached_hanging_types = to_namespace(
                AI2THOR_HANGING_OBJECT_TYPES
            )
            AI2THORDataSource._cached_attribute_state_types = to_namespace(
                AI2THOR_ATTRIBUTE_TYPES
            )
            AI2THORDataSource._cached_value_relation_custom_mapping = (
                AI2THOR_LITERAL_RELATION_MAPPING
            )
            AI2THORDataSource._cached_spatial_relation_types = to_namespace(
                AI2THOR_SPATIAL_RELATION_TYPES
            )

    def _should_include_object(self, thor_obj: Dict[str, Any]) -> bool:
        """Check if object should be included based on config policies."""
        return True

    def get_scene_by_id(self, scene_id: str) -> SceneData:
        """Get SceneData for a specific scene ID."""
        objects = []
        relationships = []
        thor_metadata = self._load_thor_room_metadata(scene_id)

        objects_metadata = thor_metadata.get("objects", [])
        # Build lookup for parent type resolution
        objects_by_id = {
            obj.get("objectId"): obj for obj in objects_metadata if obj.get("objectId")
        }
        for obj_data in objects_metadata:
            if not obj_data.get("objectId") or not self._should_include_object(
                obj_data
            ):
                continue

            oid = obj_data.get("objectId")

            # Convert to ObjectMetadata with enhanced property extraction
            object_metadata = ObjectMetadata(
                object_id=oid,
                object_type=obj_data.get("objectType", "Unknown"),
                properties=self._extract_properties(obj_data),
                position=self._extract_position(obj_data)
                if self.kg_policy.get("include_position", True)
                else None,
                rotation=self._extract_rotation(obj_data)
                if self.kg_policy.get("include_rotation", True)
                else None,
            )

            objects.append(object_metadata)

            # Room → Object
            relationships.append(
                Relationship(
                    subject_id=scene_id,
                    predicate=CorePredicates.HAS_OBJECT.value,
                    object_id=oid,
                )
            )
            relationships.append(
                Relationship(
                    subject_id=oid,
                    predicate=CorePredicates.LOCATED_AT.value,
                    object_id=scene_id,
                )
            )
            # Extract relationships based on parent receptacles
            parent_relationships = self._extract_parent_relationships(
                obj_data, objects_by_id
            )
            relationships.extend(parent_relationships)

        return SceneData(
            scene_id=scene_id, objects=objects, relationships=relationships
        )

    def get_available_scenes(self) -> List[str]:
        """Get all scene IDs from config."""
        return self.config.get("scenes", [])

    def get_scenes(self) -> Iterator[SceneData]:
        """Yield SceneData for all scenes from config."""
        scene_ids = self.get_available_scenes()
        for scene_id in scene_ids:
            yield self.get_scene_by_id(scene_id)

    def cleanup(self) -> None:
        """Clean up AI2-THOR controller with performance settings."""
        performance_config = self.config.get("performance", {})

        if performance_config.get("cleanup_between_scenes", True):
            # Force garbage collection if configured
            import gc

            gc.collect()

        if hasattr(self, "controller") and self.controller:
            self.controller.stop()

    def _extract_parent_relationships(
        self, thor_obj: Dict[str, Any], objects_by_id: Dict[str, Dict]
    ) -> List[Relationship]:
        """
        Extract spatial relationships based on parentReceptacles and parent object types.

        This is where AI2-THOR specific logic lives:
        - Container objects → "inside" relation
        - Surface objects → "onTopOf" relation
        - Hanging objects → "hanging" relation
        """
        relationships = []
        object_id = thor_obj["objectId"]
        parent_receptacles = thor_obj.get("parentReceptacles", []) or []

        for parent_id in parent_receptacles:
            parent_obj = objects_by_id.get(parent_id)
            if not parent_obj:
                continue

            parent_type = parent_obj.get("objectType", "")

            # Map parent type to spatial relation (AI2-THOR specific logic)
            relation_type = self._get_spatial_relation_for_parent(parent_type)

            if relation_type:
                relationships.append(
                    Relationship(
                        subject_id=object_id,
                        predicate=relation_type,  # Already returns string value
                        object_id=parent_id,
                    )
                )

        return relationships

    def _get_spatial_relation_for_parent(self, parent_type: str) -> str:
        """
        Map AI2-THOR object types to spatial relations.

        This encapsulates spatial relation logic based on parent object types.
        """

        if parent_type in self._cached_container_types:
            return self._cached_spatial_relation_types.inside
        elif parent_type in self._cached_surface_types:
            return self._cached_spatial_relation_types.onTopOf
        elif parent_type in self._cached_hanging_types:
            return self._cached_spatial_relation_types.hanging
        else:
            return self._cached_spatial_relation_types.near  # Default fallback relation

    def _load_thor_room_metadata(self, scene_id: str) -> Dict[str, Any]:
        """Get AI2-THOR scene metadata with agent configuration."""
        # Reset to scene
        reset_params = {"scene": scene_id}
        # Reset with configuration
        try:
            self.controller.reset(**reset_params)
        except Exception:
            self.controller.stop()
            raise Exception(f"Failed to reset AI2-THOR controller for scene {scene_id}")

        # Get scene metadata
        metadata = self.controller.last_event.metadata
        return metadata

    def _is_property_allowed(self, prop: str, kg_policy: dict) -> bool:
        policy_key = f"include_{prop}"
        return kg_policy.get(policy_key, False)

    def _extract_properties(self, thor_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract object state based on config policies."""
        properties = {}

        for field in self._cached_attribute_state_types:
            isField = AI2THOR_ATTRIBUTE_STATE_MAPPING.get(field)
            if (
                self._is_property_allowed(isField, self.kg_policy)
                and field in thor_obj
                and thor_obj.get(field, False)
            ):
                properties[field] = True

        # ToDo: Extract literlas for value relations
        for source, predicate in self._cached_value_relation_custom_mapping.items():
            if (
                self._is_property_allowed(source, self.kg_policy)
                and source in thor_obj
                and thor_obj.get(source) is not None
            ):
                if isinstance(thor_obj[source], list):
                    properties[predicate] = ", ".join(map(str, thor_obj[source]))
                else:
                    properties[predicate] = thor_obj[source]
        return properties

    def _extract_3d_vector(
        self, thor_obj: Dict[str, Any], field_name: str
    ) -> tuple[float, float, float]:
        """Generic method to extract 3D vectors (position, rotation) from AI2-THOR object."""
        vector_data = thor_obj.get(field_name)

        if vector_data is None:
            return (0.0, 0.0, 0.0)

        # AI2-THOR format is a dict with x, y, z keys
        if isinstance(vector_data, dict):
            x = vector_data.get("x", 0.0)
            y = vector_data.get("y", 0.0)
            z = vector_data.get("z", 0.0)
            return (float(x), float(y), float(z))

        # Handle list format [x, y, z]
        if isinstance(vector_data, list) and len(vector_data) >= 3:
            return (float(vector_data[0]), float(vector_data[1]), float(vector_data[2]))

        return (0.0, 0.0, 0.0)

    def _extract_position(self, thor_obj: Dict[str, Any]) -> tuple[float, float, float]:
        """Extract position from AI2-THOR object."""
        return self._extract_3d_vector(thor_obj, "position")

    def _extract_rotation(self, thor_obj: Dict[str, Any]) -> tuple[float, float, float]:
        """Extract rotation from AI2-THOR object."""
        return self._extract_3d_vector(thor_obj, "rotation")
