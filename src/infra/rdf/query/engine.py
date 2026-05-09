"""
General triple query engine for semantic search and analysis.

Works on a simple list of (s, p, o) string triples.
No rdflib dependency. Pure utility.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Union
from urllib.parse import unquote

from src.core.graph.types import Triple, TripleList, TripleSet

from typing import Pattern as RePattern

PatternLike = Union[str, RePattern[str]]

TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}

BOOLEAN_VALUES = TRUE_VALUES | FALSE_VALUES


def short_uri(x: str) -> str:
    """
    Extract local name from a URI (general helper).
    """
    x = unquote(x)
    if "#" in x:
        x = x.split("#")[-1]
    if "/" in x:
        x = x.split("/")[-1]
    return x


def entity_objtype(node: str) -> str:
    """
    Extract object type from an uri
    Works for:
    - CoffeeMachine|+01.13|...  -> CoffeeMachine
    - Table_3 -> Table
    - .../entities/Mug%7C... -> Mug
    """
    tail = short_uri(node)
    if "|" in tail:
        return tail.split("|", 1)[0]
    if "_" in tail:
        return tail.split("_", 1)[0]
    return tail


def flip_bool_object(o: object) -> str:
    ol = str(o).strip().lower()
    if ol in {"true", "1", "yes", "y", "t"}:
        return "False"
    if ol in {"false", "0", "no", "n", "f"}:
        return "True"
    return "True"


def dedupe(triples: Iterable[Triple]) -> TripleList:
    """
    Remove duplicate triples preserving order.
    """
    seen: Set[Triple] = set()
    out: TripleList = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass
class TripleQueryEngine:
    """
    Query & grouping utilities for triples.

    This engine is agnostic to any domain schema; it only operates on triples.
    """

    triples: TripleList

    def __post_init__(self) -> None:
        self.triples = dedupe(self.triples)

    # ---------------- Basic Queries ----------------

    def query(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> TripleList:
        """
        Filter triples by exact match on subject/predicate/object.
        Any field can be None to mean "match all".
        """
        out: TripleList = []
        for triple in self.triples:
            if (
                (subject is None or triple.s == subject)
                and (predicate is None or triple.p == predicate)
                and (obj is None or triple.o == obj)
            ):
                out.append(triple)
        return out

    def group_by(self, field: str) -> Dict[str, TripleList]:
        """
        Group triples by one of: subject | predicate | object
        """
        if field not in ["subject", "predicate", "object"]:
            raise ValueError("field must be one of: subject, predicate, object")

        grouped: DefaultDict[str, TripleList] = defaultdict(list)
        for triple in self.triples:
            if field == "subject":
                key = triple.s
            elif field == "predicate":
                key = triple.p
            else:  # object
                key = triple.o
            grouped[key].append(triple)
        return dict(grouped)

    # ---------------- Relationship Queries ----------------

    def objects_of(self, subject: str, predicates: List[str]) -> List[str]:
        return [
            triple.o
            for triple in self.triples
            if triple.s == subject and triple.p in predicates
        ]

    def subjects_of(self, obj: str, predicates: List[str]) -> List[str]:
        return [
            triple.s
            for triple in self.triples
            if triple.o == obj and triple.p in predicates
        ]

    def nodes_of(self, predicates: List[str]) -> Set[str]:
        nodes: Set[str] = set()
        for triple in self.triples:
            if triple.p in predicates:
                nodes.add(triple.s)
                nodes.add(triple.o)
        return nodes

    def has_triple(self, subject=None, predicate=None, obj=None) -> bool:
        for triple in self.triples:
            if (
                (subject is None or triple.s == subject)
                and (predicate is None or triple.p == predicate)
                and (obj is None or triple.o == obj)
            ):
                return True
        return False

    def is_boolean_predicate(self, subject: str, predicate: str) -> bool:

        for triple in self.triples:
            if triple.s == subject and triple.p == predicate:
                val = str(triple.o).strip().lower()

                if val in BOOLEAN_VALUES:
                    return True

        return False

    # ---------------- Pattern Grouping ----------------

    def group_by_namespace(self, namespace: str) -> Dict[str, TripleList]:
        """
        Group triples by namespace name in subject or object URIs using two-phase mapping.

        Phase 1: Build object-to-context mapping via hasObject/inScene relations
        Phase 2: Group all triples using this mapping (similar to group_by_floorplan)

        Args:
            namespace: Either a full namespace URI (e.g., "http://example.com/contexts/")
                      or just a namespace name (e.g., "contexts")
        """
        from urllib.parse import unquote

        # Helper functions
        def _short(x: str) -> str:
            x = unquote(x)
            if "#" in x:
                x = x.split("#")[-1]
            if "/" in x:
                x = x.split("/")[-1]
            return x

        def _extract_context(uri: str, ns: str) -> str | None:
            u = unquote(uri)
            # If namespace looks like full URI, use exact matching
            if ns.startswith(("http://", "https://", "file://")):
                if ns in u:
                    parts = u.split(ns, 1)
                    if len(parts) > 1:
                        return parts[1].split("/", 1)[0].split("#", 1)[0]
            else:
                # Use regex to match namespace name
                pattern = re.compile(rf"{re.escape(ns)}[/#]([^/#]+)")
                match = pattern.search(u)
                if match:
                    return match.group(1)
            return None

        # Phase 1: Build object-to-context mapping
        obj2context: Dict[str, str] = {}
        for triple in self.triples:
            pred = _short(triple.p).lower()

            if pred == "hasobject":
                context = _extract_context(triple.s, namespace)
                if context:
                    obj2context[triple.o] = context

            elif pred == "inscene":
                context = _extract_context(triple.o, namespace)
                if context:
                    obj2context[triple.s] = context

        # Phase 2: Group all triples using the mapping
        grouped: DefaultDict[str, TripleList] = defaultdict(list)
        for triple in self.triples:
            # Try direct context extraction first
            context = _extract_context(triple.s, namespace) or _extract_context(
                triple.o, namespace
            )

            # If not directly found, use the mapping
            if not context:
                context = obj2context.get(triple.s) or obj2context.get(triple.o)

            if context:
                grouped[context].append(triple)

        return dict(grouped)

    def group_by_pattern(
        self,
        *,
        entity_pattern: PatternLike,
        predicates: List[str] = [],
        match_on: str = "subject",
        key_by: str = "entity",  # "entity" | "predicate"
    ) -> Dict[str, TripleList]:
        """
        Group triples where subject/object matches a pattern and predicate is in `predicates`.

        entity_pattern:
          - string => substring match (escaped)
          - regex pattern => regex search

        match_on:
          - "subject" or "object"

        key_by:
          - "entity" => group key is matched subject/object
          - "predicate" => group key is predicate
        """
        if match_on not in {"subject", "object"}:
            raise ValueError("match_on must be 'subject' or 'object'")
        if key_by not in {"entity", "predicate"}:
            raise ValueError("key_by must be 'entity' or 'predicate'")

        pattern = (
            re.compile(re.escape(entity_pattern))
            if isinstance(entity_pattern, str)
            else entity_pattern
        )

        grouped: DefaultDict[str, TripleList] = defaultdict(list)

        for triple in self.triples:
            if predicates:
                if triple.p not in predicates:
                    continue

            target = triple.s if match_on == "subject" else triple.o
            if not pattern.search(target):
                continue

            key = target if key_by == "entity" else triple.p
            grouped[key].append(triple)

        return dict(grouped)

    # ---------------- Entity Resolution / Grouping ----------------

    def _extract_by_pattern(self, uri: str, pattern: RePattern[str]) -> Optional[str]:
        u = unquote(uri)
        m = pattern.search(u)
        return m.group(0) if m else None

    def build_entity_map(
        self,
        *,
        entity_pattern: RePattern[str],
        forward_rel: str,
        backward_rel: str,
        predicate_shortener=short_uri,
    ) -> Dict[str, str]:
        """
        Build a mapping from node -> entity_id based on forward/backward relations.

        Example use case:
          - forward_rel="hasId":  entity appears in subject
          - backward_rel="idOf":  entity appears in object

        predicate_shortener lets you normalize URIs to local names (default: short_uri).
        """
        mapping: Dict[str, str] = {}

        f = forward_rel.lower()
        b = backward_rel.lower()

        for triple in self.triples:
            pred = predicate_shortener(triple.p).lower()

            if pred == f:
                ent = self._extract_by_pattern(triple.s, entity_pattern)
                if ent:
                    mapping[triple.o] = ent

            elif pred == b:
                ent = self._extract_by_pattern(triple.o, entity_pattern)
                if ent:
                    mapping[triple.s] = ent

        return mapping

    def group_by_entity(
        self,
        *,
        entity_pattern: RePattern[str],
        forward_rel: str,
        backward_rel: str,
        default: str = "UNKNOWN",
    ) -> Dict[str, TripleList]:
        """
        Group all triples by a resolved entity id.
        """
        entity_map = self.build_entity_map(
            entity_pattern=entity_pattern,
            forward_rel=forward_rel,
            backward_rel=backward_rel,
        )

        grouped: DefaultDict[str, TripleList] = defaultdict(list)

        for triple in self.triples:
            ent = self._extract_by_pattern(
                triple.s, entity_pattern
            ) or self._extract_by_pattern(triple.o, entity_pattern)
            if not ent:
                ent = entity_map.get(triple.s) or entity_map.get(triple.o) or default
            grouped[ent].append(triple)

        return dict(grouped)

    # ---------------- Convenience ----------------

    def as_set(self) -> TripleSet:
        return set(self.triples)

    def set_triples(self, triples: Iterable[Triple]) -> None:
        self.triples = dedupe(triples)
