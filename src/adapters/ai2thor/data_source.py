from __future__ import annotations

from typing import Any, Dict, Iterator, List
from ai2thor.controller import Controller

from src.adapters.ai2thor.result import SceneRandomizerSummary
from src.adapters.ai2thor.scene_randomizer import SceneRandomizer
from src.core.graph.types import Graph as SceneGraph, Object, Relationship
from src.ports.graph_data_source import GraphDataSource
from src.utils.exceptions import ConfigurationError, DataSourceError

from src.adapters.ai2thor.config import AI2ThorDataSourceConfig
from src.adapters.ai2thor.registry.entities import CONTAINER_ROLES, SURFACE_ROLES, HANGING_ROLES
from src.adapters.ai2thor.semantic_rules import build_semantic_map
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---- canonical predicate ids (ontology will map later) ----
HAS_OBJECT = "hasObject"
IN_SCENE = "inScene"

# ---- AI2-THOR metadata keys ----
OBJECTS = "objects"
OBJECT_ID = "objectId"
OBJECT_TYPE = "objectType"
PARENT_RECEPTACLES = "parentReceptacles"
POSITION = "position"
ROTATION = "rotation"

# ---- spatial predicate ids ----
INSIDE = "inside"
ON_TOP_OF = "onTopOf"
HANGING = "hanging"
NEAR = "near"

UNKNOWN = "Unknown"

# metadata fields we never treat as KG properties
NOT_CONSIDER_PROPERTY = {OBJECT_ID, OBJECT_TYPE, PARENT_RECEPTACLES, POSITION, ROTATION}


class AI2THORDataSource(GraphDataSource):
    """
    AI2-THOR data source (adapter).
    Produces SceneGraph using predicate ids that ontology will map later.
    """

    def __init__(self, config: AI2ThorDataSourceConfig):
        self.config = config
        self.kg_policy = config.knowledge_graph_policy
        self.controller = self._create_controller(config)

    def _create_controller(self, cfg: AI2ThorDataSourceConfig) -> Controller:
        try:
            return Controller(
                **cfg.controller.settings,
                renderImage=cfg.controller.render_image,
                renderDepthImage=cfg.controller.render_depth,
                renderInstanceSegmentation=cfg.controller.render_instance_segmentation,
                renderSemanticSegmentation=cfg.controller.render_semantic_segmentation,
                visibilityDistance=cfg.controller.visibility_distance,
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create AI2-THOR controller: {e}") from e

    # ---------------- GraphDataSource API ----------------

    def get_available_graph_ids(self) -> List[str]:
        return list(self.config.scenes)

    def get_graphs(self) -> Iterator[SceneGraph]:
        for sid in self.config.scenes:
            yield self.get_graph_by_id(sid)

    def get_graph_by_id(self, graph_id: str) -> SceneGraph:
        try:
            metadata = self._load_thor_room_metadata(graph_id)
        except Exception as e:
            raise DataSourceError(f"Failed to load AI2-THOR scene {graph_id}: {e}") from e
        objects_meta = metadata.get(OBJECTS, []) or []
        objects_by_id = {o.get(OBJECT_ID): o for o in objects_meta if o.get(OBJECT_ID)}

        objects: List[Object] = []
        rels: List[Relationship] = []

        for o in objects_meta:
            oid = o.get(OBJECT_ID)
            if not oid:
                continue

            obj_type = o.get(OBJECT_TYPE, UNKNOWN)

            objects.append(
                Object(
                    object_id=oid,
                    object_type=obj_type,
                    properties=self._extract_properties(o),
                    position=self._extract_position(o) if self.kg_policy.get("include_position", False) else None,
                    rotation=self._extract_rotation(o) if self.kg_policy.get("include_rotation", False) else None,
                )
            )

            # scene membership (bidirectional)
            rels.append(Relationship(subject_id=graph_id, predicate=HAS_OBJECT, object_id=oid))
            rels.append(Relationship(subject_id=oid, predicate=IN_SCENE, object_id=graph_id))

            # parent-based spatial relations
            rels.extend(self._extract_parent_relationships(o, objects_by_id))

        return SceneGraph(graph_id=graph_id, objects=objects, relationships=rels, metadata=None)

    def cleanup(self) -> None:
        perf = self.config.performance or {}

        if perf.get("cleanup_between_graphs", True):
            import gc
            gc.collect()

        if getattr(self, "controller", None):
            try:
                self.controller.stop()
            except Exception:
                pass

    # ---------------- AI2-THOR specifics ----------------

    def _load_thor_room_metadata(self, graph_id: str) -> Dict[str, Any]:
        try:
            self.controller.reset(scene=graph_id)
            randomizer = SceneRandomizer(
                            controller=self.controller,
                            scene_id=graph_id,
                            seed=123,
                            randomize_receptacles=True,
                            max_objects_per_receptacle=2,
                            use_semantic_rules=True,
                        )
            

            logger.info("-----"*20)
            logger.info(f"Scene {graph_id}")
            logger.info("Initial semantic map:")
            logger.info(build_semantic_map(controller=self.controller))
            logger.info("-----"*20)
            result = randomizer.randomize()
            logger.info("Final semantic map after randomization:")
            SceneRandomizerSummary(results=[result]).print_summary()
            logger.info(build_semantic_map(controller=self.controller))
            logger.info("-----"*20)

            return self.controller.last_event.metadata
        except Exception as e:
            try:
                self.controller.stop()
            except Exception:
                pass
            raise DataSourceError(f"Controller reset failed for graph {graph_id}: {e}") from e

    def _extract_parent_relationships(
        self, thor_obj: Dict[str, Any], objects_by_id: Dict[str, Dict[str, Any]]
    ) -> List[Relationship]:
        rels: List[Relationship] = []
        oid = thor_obj[OBJECT_ID]
        parents = thor_obj.get(PARENT_RECEPTACLES) or []

        for pid in parents:
            pobj = objects_by_id.get(pid)
            if not pobj:
                continue

            ptype = pobj.get(OBJECT_TYPE, "")
            pred = self._spatial_predicate_for_parent(ptype)
            rels.append(Relationship(subject_id=oid, predicate=pred, object_id=pid))

        return rels

    def _spatial_predicate_for_parent(self, parent_type: str) -> str:
        if parent_type in CONTAINER_ROLES:
            return INSIDE
        if parent_type in SURFACE_ROLES:
            return ON_TOP_OF
        if parent_type in HANGING_ROLES:
            return HANGING
        return NEAR

    # ---------------- Property extraction ----------------

    def _is_property_allowed(self, prop: str) -> bool:
        if prop in NOT_CONSIDER_PROPERTY:
            return False
        # policy keys like include_isOpen, include_mass, include_temperature...
        return bool(self.kg_policy.get(f"include_{prop}", False))


    def _extract_properties(self, thor_obj: Dict[str, Any]) -> Dict[str, Any]:
        props: Dict[str, Any] = {}

        for prop, value in thor_obj.items():
            if not self._is_property_allowed(prop):
                continue
            if value is None:
                continue

            if isinstance(value, bool):
                props[prop] = value
            else:
                props[prop] = ", ".join(map(str, value)) if isinstance(value, list) else value

        return props

    # ---------------- vectors ----------------

    @staticmethod
    def _extract_3d_vector(thor_obj: Dict[str, Any], field: str) -> tuple[float, float, float]:
        v = thor_obj.get(field)
        if isinstance(v, dict):
            return (float(v.get("x", 0.0)), float(v.get("y", 0.0)), float(v.get("z", 0.0)))
        if isinstance(v, list) and len(v) >= 3:
            return (float(v[0]), float(v[1]), float(v[2]))
        return (0.0, 0.0, 0.0)

    def _extract_position(self, thor_obj: Dict[str, Any]) -> tuple[float, float, float]:
        return self._extract_3d_vector(thor_obj, POSITION)

    def _extract_rotation(self, thor_obj: Dict[str, Any]) -> tuple[float, float, float]:
        return self._extract_3d_vector(thor_obj, ROTATION)
