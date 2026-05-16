from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.exceptions import ConfigurationError
from src.adapters.ai2thor.scene.config import (
    AI2ThorDataSourceConfig,
    AI2ThorControllerConfig,
)


def load_ai2thor_config(path: str | Path) -> AI2ThorDataSourceConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Config file not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parse failed: {p}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Invalid config format in {p}: expected dict")

    # AI2THOR settings live under the `ai2thor:` key.
    ai2thor = raw.get("ai2thor")
    if not isinstance(ai2thor, dict):
        raise ConfigurationError(
            f"Config {p} must have an 'ai2thor:' section with at least a 'scenes' list"
        )

    scenes = ai2thor.get("scenes")
    if not scenes or not isinstance(scenes, list):
        raise ConfigurationError("ai2thor.scenes must be a non-empty list")

    controller_raw: Dict[str, Any] = ai2thor.get("controller", {}) or {}
    kg_policy: Dict[str, bool] = ai2thor.get("knowledge_graph_policy", {}) or {}
    performance: Dict[str, Any] = ai2thor.get("performance", {}) or {}

    # keep only actual controller kwargs (strip comments, None)
    controller_settings = {
        k: v
        for k, v in controller_raw.items()
        if v is not None and not str(k).startswith("#")
    }

    return AI2ThorDataSourceConfig(
        scenes=[str(s) for s in scenes],
        controller=AI2ThorControllerConfig(
            settings=controller_settings,
            visibility_distance=float(controller_raw.get("visibilityDistance", 1.5)),
            render_image=bool(controller_raw.get("renderImage", False)),
            render_depth=bool(controller_raw.get("renderDepthImage", False)),
            render_instance_segmentation=bool(
                controller_raw.get("renderInstanceSegmentation", False)
            ),
            render_semantic_segmentation=bool(
                controller_raw.get("renderSemanticSegmentation", False)
            ),
        ),
        knowledge_graph_policy={str(k): bool(v) for k, v in kg_policy.items()},
        performance=performance,
    )
