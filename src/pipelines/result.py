# src/core/pipeline/result.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PipelineRunResult:
    success: bool
    out_path: str
    format: str
    num_objects: int
    num_relations: int
    total_triples: int
    contexts_processed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
