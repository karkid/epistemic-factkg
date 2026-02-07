import random
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.adapters.ai2thor.semantic_rules import get_preferred_receptacles
from src.adapters.ai2thor.result import SceneRandomizerResult, ObjectPlacement, ObjectStateChange
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SceneRandomizer:
    """
    Handles scene-level randomization for AI2-THOR environments.
    Physics-safe, throttled, and with robust logging.
    """

    def __init__(
        self,
        controller,
        scene_id: str,
        seed: int | None = None,
        randomize_agent: bool = False,
        randomize_states: bool = True,
        randomize_objects: bool = False,
        randomize_receptacles: bool = True,
        max_objects_per_receptacle: int = 4,
        use_semantic_rules: bool = False,
        randomize_lighting: bool = False,
        use_physics_random_spawn: bool = False,
    ):
        self.controller = controller
        self.scene_id = scene_id
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.randomize_agent = randomize_agent
        self.randomize_states = randomize_states
        self.randomize_objects = randomize_objects
        self.randomize_receptacles = randomize_receptacles
        self.max_objects_per_receptacle = max_objects_per_receptacle
        self.randomize_lighting = randomize_lighting
        self.use_physics_random_spawn = use_physics_random_spawn
        self.use_semantic_rules = use_semantic_rules

        # Scene result tracking
        self.result = SceneRandomizerResult(scene_id=scene_id)

    # ----------------------------
    # Public API
    # ----------------------------
    def randomize(self) -> SceneRandomizerResult:
        """Apply all enabled randomizations with throttling to prevent physics freeze."""
        start = time.time()
        logger.info(f"===== Scene Randomization Started for {self.scene_id} =====")

        try:
            if self.use_physics_random_spawn:
                logger.info("Initial random spawn")
                self._safe_step(action="InitialRandomSpawn")
                self._wait_for_physics(20)

            if self.randomize_objects:
                logger.info("Randomizing object positions")
                self._randomize_objects()
                self._wait_for_physics(20)

            if self.randomize_states:
                logger.info("Randomizing object states")
                self._randomize_states()
                self._wait_for_physics(20)

            if self.randomize_agent:
                logger.info("Randomizing agent position (skipped teleport)")
                self._randomize_agent()
                self._wait_for_physics(20)

            if self.randomize_lighting:
                logger.info("Randomizing lighting")
                self._randomize_lighting()
                self._wait_for_physics(20)

            if self.randomize_receptacles:
                logger.info("Randomizing receptacle placement")
                self._randomize_receptacles()
                self._wait_for_physics(20)

        finally:
            elapsed = time.time() - start
            logger.info(f"Scene randomization finished in {elapsed:.1f}s")

        return self.result

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _wait_for_physics(self, steps=10):
        """Let Unity physics settle to avoid freezes."""
        for _ in range(steps):
            self._safe_step(action="Pass")

    def _safe_step(self, **kwargs):
        """Safe wrapper for controller.step() with logging and timeout detection."""
        start = time.time()
        try:
            event = self.controller.step(**kwargs)
        except Exception as e:
            logger.error(f"Step failed: {kwargs.get('action', '')}, Error: {e}")
            return None

        if time.time() - start > 5:
            logger.warning(f"Step {kwargs.get('action', '')} took >5s, possible Unity freeze")

        return event

    def _get_objects(self):
        return getattr(self.controller.last_event.metadata, "objects", []) \
            or self.controller.last_event.metadata.get("objects", [])

    # ----------------------------
    # Randomizations
    # ----------------------------
    def _randomize_agent(self):
        # Skip teleport to avoid scene freeze
        logger.info("Skipping agent teleport to avoid physics freeze")

    def _randomize_objects(self):
        objects = self._get_objects()
        event = self._safe_step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn", []) if event else []

        if not positions:
            return

        for obj in objects:
            if not obj.get("pickupable", False):
                continue

            from_pos = obj.get("position")
            to_pos = random.choice(positions)

            logger.info(f"Placing {obj['objectId']} at {to_pos}")
            success = self._safe_step(
                action="PlaceObjectAtPoint",
                objectId=obj["objectId"],
                position=to_pos
            )

            self.result.object_placements.append(
                ObjectPlacement(
                    object_id=obj["objectId"],
                    object_type=obj["objectType"],
                    from_position=from_pos,
                    to_position=to_pos,
                    success=bool(success),
                )
            )

    def _randomize_states(self):
        objects = self._get_objects()
        # Only consider 30% to avoid physics overload
        candidates = [o for o in objects if o.get("openable") or o.get("toggleable")]
        random.shuffle(candidates)
        limit = max(1, int(len(candidates) * 0.3))

        for obj in candidates[:limit]:
            oid = obj["objectId"]
            old_state = []
            new_state = []

            if obj.get("openable", False):
                old_state.append("Open" if obj.get("isOpen", False) else "Closed")
                action = "OpenObject" if random.random() < 0.5 else "CloseObject"
                event = self._safe_step(action=action, objectId=oid, forceAction=True)
                new_state.append("Open" if action == "OpenObject" else "Closed")
                self._wait_for_physics(3)  # small pause after each open/close

            if obj.get("toggleable", False):
                old_state.append("On" if obj.get("isToggled", False) else "Off")
                action = "ToggleObjectOn" if random.random() < 0.5 else "ToggleObjectOff"
                event = self._safe_step(action=action, objectId=oid, forceAction=True)
                new_state.append("On" if action == "ToggleObjectOn" else "Off")
                self._wait_for_physics(3)  # small pause after each toggle

            self.result.state_changes.append(
                ObjectStateChange(
                    object_id=oid,
                    object_type=obj["objectType"],
                    from_state="|".join(old_state) if old_state else "None",
                    to_state="|".join(new_state) if new_state else "None",
                    success=True
                )
            )


    def _randomize_lighting(self):
        self._safe_step(action="RandomizeLighting")

    def _randomize_receptacles(self):
        objects = self._get_objects()
        items = [o for o in objects if o.get("pickupable") and not o.get("isPickedUp", False)]
        receptacles = [o for o in objects if o.get("receptacle", False)]

        if not items or not receptacles:
            return

        usage = {r["objectId"]: 0 for r in receptacles}
        random.shuffle(items)

        for item in items:
            obj_type = item["objectType"]
            preferred = get_preferred_receptacles(obj_type) if self.use_semantic_rules else []

            candidates = [r for r in receptacles if r["objectType"] in preferred]
            if not candidates:
                candidates = receptacles
            random.shuffle(candidates)

            for rec in candidates:
                if usage[rec["objectId"]] >= self.max_objects_per_receptacle:
                    continue

                logger.info(f"Placing {item['objectId']} → {rec['objectId']}")
                success = self._place_in_receptacle(item, rec)

                if success:
                    usage[rec["objectId"]] += 1
                    self.result.receptacles_used = sum(usage.values())
                    break

    def _place_in_receptacle(self, item, receptacle):
        """Physics-safe placement: pickup, move slightly forward, drop"""
        ctrl = self.controller

        # Skip teleport to avoid freeze
        logger.debug(f"Placing {item['objectId']} near {receptacle['objectId']}")

        # Open receptacle if needed
        if receptacle.get("openable", False) and not receptacle.get("isOpen", False):
            event = self._safe_step(
                action="OpenObject",
                objectId=receptacle["objectId"],
                forceAction=True,
            )
            if not event or not event.metadata.get("lastActionSuccess", False):
                return False

        # Pickup item
        event = self._safe_step(
            action="PickupObject",
            objectId=item["objectId"],
            forceAction=True,
        )
        if not event or not event.metadata.get("lastActionSuccess", False):
            return False

        # Move slightly forward for alignment
        self._safe_step(action="MoveAhead", moveMagnitude=0.05)

        # Drop object
        event = self._safe_step(
            action="PutObject",
            forceAction=True,
        )
        return event and event.metadata.get("lastActionSuccess", False)
