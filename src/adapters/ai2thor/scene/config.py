from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class AI2ThorControllerConfig:
    settings: Dict[str, Any] = field(default_factory=dict)
    visibility_distance: float = 1.5
    render_image: bool = False
    render_depth: bool = False
    render_instance_segmentation: bool = False
    render_semantic_segmentation: bool = False


@dataclass(frozen=True)
class AI2ThorDataSourceConfig:
    scenes: List[str]
    controller: AI2ThorControllerConfig = field(default_factory=AI2ThorControllerConfig)
    knowledge_graph_policy: Dict[str, bool] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    randomizer: Dict[str, Any] = field(default_factory=dict)
