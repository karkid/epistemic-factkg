from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------
# Event-level dataclasses
# ----------------------------
@dataclass
class ObjectPlacement:
    object_id: str
    object_type: str
    from_position: Optional[Tuple[float, float, float]] = None
    to_position: Optional[Tuple[float, float, float]] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ObjectStateChange:
    object_id: str
    object_type: str
    from_state: str
    to_state: str
    success: bool = True
    error: Optional[str] = None


# ----------------------------
# Scene-level result
# ----------------------------
@dataclass
class SceneRandomizerResult:
    scene_id: str
    object_placements: List[ObjectPlacement] = field(default_factory=list)
    state_changes: List[ObjectStateChange] = field(default_factory=list)
    receptacles_used: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ----------------------------
# Summary across scenes
# ----------------------------
@dataclass
class SceneRandomizerSummary:
    results: List[SceneRandomizerResult] = field(default_factory=list)

    def __init__(self, results: Optional[List[SceneRandomizerResult]] = None):
        self.results = results if results is not None else []

    def add_result(self, result: SceneRandomizerResult):
        self.results.append(result)

    def print_summary(self, logger_func=logger.info):
        logger_func("=" * 72)
        logger_func("Scene Randomization Summary")
        logger_func("=" * 72)

        for res in self.results:
            logger_func(f"Scene ID: {res.scene_id}")
            logger_func(f"Receptacles used: {res.receptacles_used}")
            logger_func(f"Object Placements ({len(res.object_placements)}):")
            for obj in res.object_placements:
                status = "SUCCESS" if obj.success else f"FAILED ({obj.error})"
                logger_func(
                    f"  {obj.object_type}|{obj.object_id} "
                    f"{obj.from_position} -> {obj.to_position} [{status}]"
                )

            logger_func(f"State Changes ({len(res.state_changes)}):")
            for state in res.state_changes:
                status = "SUCCESS" if state.success else f"FAILED ({state.error})"
                logger_func(
                    f"  {state.object_type}|{state.object_id} "
                    f"{state.from_state} -> {state.to_state} [{status}]"
                )

            if res.warnings:
                logger_func(f"Warnings: {len(res.warnings)}")
                for w in res.warnings:
                    logger_func(f"  - {w}")
            if res.errors:
                logger_func(f"Errors: {len(res.errors)}")
                for e in res.errors:
                    logger_func(f"  - {e}")

            logger_func("-" * 72)
        # ----------------------------
        # Overall totals
        # ----------------------------
        total_objects = sum(len(r.object_placements) for r in self.results)
        total_states = sum(len(r.state_changes) for r in self.results)
        total_receptacles = sum(r.receptacles_used for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)
        total_errors = sum(len(r.errors) for r in self.results)

        logger_func("=" * 72)
        logger_func(f"Scenes processed : {len(self.results)}")
        logger_func(f"Total object placements : {total_objects}")
        logger_func(f"Total state changes     : {total_states}")
        logger_func(f"Total receptacles used  : {total_receptacles}")
        logger_func(f"Total warnings          : {total_warnings}")
        logger_func(f"Total errors            : {total_errors}")
        logger_func("=" * 72)
