"""Graph type primitives shared across RDF, NLG, and adapter layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, NewType, Union

IRI = NewType("IRI", str)
NodeId = NewType("NodeId", str)

Literal = Union[str, int, float, bool]
Term = Union[IRI, NodeId, Literal]


@dataclass(frozen=True, slots=True)
class Triple:
    s: str
    p: str
    o: Term

    def __iter__(self):
        yield self.s
        yield self.p
        yield self.o


TripleList = list[Triple]
TripleSet = set[Triple]


@dataclass(frozen=True, slots=True)
class Quad:
    s: str
    p: str
    o: Term
    g: str  # graph/source/context id


def is_literal(x: Term) -> bool:
    return isinstance(x, (str, int, float, bool))


def is_iri(x: str) -> bool:
    return x.startswith("http://") or x.startswith("https://")


@dataclass(frozen=True)
class Relationship:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Object:
    object_id: str
    object_type: str
    properties: Dict[str, Any]
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None


@dataclass
class Graph:
    graph_id: str
    objects: List[Object]
    relationships: List[Relationship] = None
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
