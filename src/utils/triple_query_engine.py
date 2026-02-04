"""
Triple Query Engine for semantic search and analysis.

This module provides a flexible query engine for working with RDF triples,
supporting pattern matching, entity grouping, and semantic analysis operations.
"""

from typing import List, Dict, Set, Tuple, Pattern, Union
from collections import defaultdict
from rdflib import Graph, URIRef, Literal
import re
from urllib.parse import unquote

Triple = Tuple[str, str, str]
Pattern = Union[str, Pattern]


class TripleQueryEngine:
    """
    Flexible query engine for RDF triples with pattern matching and grouping capabilities.

    This engine provides methods to load, query, and analyze RDF data with support for:
    - Basic triple queries (subject, predicate, object filtering)
    - Pattern-based entity extraction and grouping
    - Relationship analysis and graph traversal
    - Performance-optimized duplicate filtering

    Example:
        engine = TripleQueryEngine()
        engine.load_from_ttl('knowledge_graph.ttl')
        results = engine.query(predicate='hasObject')
    """

    def __init__(self, data_source: List[Triple] | None = None):
        """
        Initialize engine with triples.
        """
        self.data_source = self.filter_duplicates(data_source or [])

    # ------------------ Utilities ------------------

    @staticmethod
    def short(x: str) -> str:
        """Extract local name from URI."""
        x = unquote(x)

        if "#" in x:
            x = x.split("#")[-1]

        if "/" in x:
            x = x.split("/")[-1]

        return x

    @staticmethod
    def filter_duplicates(triples: List[Triple]) -> List[Triple]:
        """Remove duplicate triples."""
        seen = set()
        out = []

        for t in triples:
            if t not in seen:
                seen.add(t)
                out.append(t)

        return out

    def _term_to_str(self, t) -> str:
        if isinstance(t, URIRef):
            return str(t)

        if isinstance(t, Literal):
            return str(t.toPython())

        return str(t)

    def _extract_by_pattern(self, uri: str, pattern: Pattern[str]) -> str | None:
        """Extract entity using regex."""
        u = unquote(uri)
        m = pattern.search(u)

        return m.group(0) if m else None

    # ------------------ Loaders ------------------

    def load_from_ttl(self, ttl_path: str) -> None:
        """Load triples from Turtle file."""
        g = Graph()
        g.parse(ttl_path, format="turtle")

        triples = []

        for s, p, o in g:
            triples.append(
                (self._term_to_str(s), self._term_to_str(p), self._term_to_str(o))
            )

        self.data_source = self.filter_duplicates(triples)

    @classmethod
    def from_ttl(cls, ttl_path: str) -> "TripleQueryEngine":
        """Factory constructor."""
        engine = cls()
        engine.load_from_ttl(ttl_path)
        return engine

    def set_data_source(self, triples: List[Triple]) -> None:
        """Set data source triples."""
        self.data_source = self.filter_duplicates(triples)

    # ------------------ Basic Queries ------------------

    def query(self, subject=None, predicate=None, obj=None) -> List[Triple]:

        results = []

        for s, p, o in self.data_source:
            if (
                (subject is None or s == subject)
                and (predicate is None or p == predicate)
                and (obj is None or o == obj)
            ):
                results.append((s, p, o))

        return results

    def group_by(self, field: str) -> Dict[str, List[Triple]]:
        """
        Group by: subject | predicate | object
        """
        idx = {"subject": 0, "predicate": 1, "object": 2}

        if field not in idx:
            raise ValueError("field must be subject/predicate/object")

        i = idx[field]

        grouped = defaultdict(list)

        for t in self.data_source:
            grouped[t[i]].append(t)

        return dict(grouped)

    # ------------------ Relationship Queries ------------------

    def objects_of(self, subject: str, predicates: List[str]) -> List[str]:

        return [o for s, p, o in self.data_source if s == subject and p in predicates]

    def subjects_of(self, obj: str, predicates: List[str]) -> List[str]:

        return [s for s, p, o in self.data_source if o == obj and p in predicates]

    def nodes_of(self, predicates: List[str]) -> Set[str]:

        nodes = set()

        for s, p, o in self.data_source:
            if p in predicates:
                nodes.add(s)
                nodes.add(o)

        return nodes

    # ------------------ Pattern Grouping ------------------

    def group_by_pattern(
        self,
        entity_pattern: Pattern[str] | str,
        predicates: List[str],
        match_on: str = "subject",
        group_by: str = "entity",
    ) -> Dict[str, List[Triple]]:

        if isinstance(entity_pattern, str):
            pattern = re.compile(re.escape(entity_pattern))
        else:
            pattern = entity_pattern

        grouped = defaultdict(list)

        for s, p, o in self.data_source:
            if p not in predicates:
                continue

            target = s if match_on == "subject" else o

            if not pattern.search(target):
                continue

            key = target if group_by == "entity" else p

            grouped[key].append((s, p, o))

        return dict(grouped)

    # ------------------ Entity Resolution ------------------

    def build_entity_map(
        self, entity_pattern: Pattern[str], forward_rel: str, backward_rel: str
    ) -> Dict[str, str]:

        mapping = {}

        for s, p, o in self.data_source:
            pred = self.short(p).lower()

            if pred == forward_rel.lower():
                ent = self._extract_by_pattern(s, entity_pattern)

                if ent:
                    mapping[o] = ent

            elif pred == backward_rel.lower():
                ent = self._extract_by_pattern(o, entity_pattern)

                if ent:
                    mapping[s] = ent

        return mapping

    def group_by_entity(
        self,
        entity_pattern: Pattern[str],
        forward_rel: str,
        backward_rel: str,
        default: str = "UNKNOWN",
    ) -> Dict[str, List[Triple]]:

        entity_map = self.build_entity_map(entity_pattern, forward_rel, backward_rel)

        grouped = defaultdict(list)

        for s, p, o in self.data_source:
            ent = self._extract_by_pattern(
                s, entity_pattern
            ) or self._extract_by_pattern(o, entity_pattern)

            if not ent:
                ent = entity_map.get(s) or entity_map.get(o) or default

            grouped[ent].append((s, p, o))

        return dict(grouped)
