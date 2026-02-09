# src/core/build/result.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List
from rdflib import Graph as RDFGraph


@dataclass
class GraphBuildResult:
    graph: RDFGraph
    num_objects: int
    num_relations: int
    contexts_processed: Set[str]
    warnings: List[str] | None = None
    errors: List[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_triples(self) -> int:
        return len(self.graph)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)