"""
Clean data source interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, NewType, Union
from urllib.parse import unquote

IRI = NewType("IRI", str)
NodeId = NewType("NodeId", str)

Literal = Union[str, int, float, bool]
Term = Union[IRI, NodeId, Literal]


@dataclass(frozen=True, slots=True)
class Triple:

    s: str          # or IRI/NodeId
    p: str          # or IRI
    o: Term       # union that supports literals

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

    # NOTE: bool is a subclass of int, but that’s fine here.
    return isinstance(x, (str, int, float, bool))


def is_iri(x: str) -> bool:
    
    return x.startswith("http://") or x.startswith("https://")

@dataclass(frozen=True)
class Relationship():
    """Relationship between two objects."""

    subject_id: str  # Object that has the relationship
    predicate: str  # Type of relationship (e.g., "isOn", "contains", "nextTo")
    object_id: str  # Object being related to
    confidence: float = 1.0  # Optional confidence score

@dataclass(frozen=True)
class Object():
    """Metadata for a single object in a scene."""

    object_id: str
    object_type: str
    properties: Dict[str, Any]  # Object properties (color, material, etc.)
    position: tuple[float, float, float] | None = None
    rotation: tuple[float, float, float] | None = None


@dataclass
class Graph():
    """Complete data for a scene."""

    graph_id: str
    objects: List[Object]
    relationships: List[Relationship] = None  # Object-to-object relationships
    metadata: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
