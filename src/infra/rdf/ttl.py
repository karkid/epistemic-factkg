"""
RDF/Turtle IO helpers for working with triples.

This module is intentionally small and isolated because it depends on rdflib.
Keep rdflib out of core graph/query logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rdflib import Graph as RDFGraph, URIRef, Literal

from src.core.graph.types import Triple, TripleList


def _term_to_str(t) -> str:
    """
    Convert rdflib terms to plain Python strings.
    - URIRef -> str(uri)
    - Literal -> python value -> str(value)
    - fallback -> str(...)
    """
    if isinstance(t, URIRef):
        return str(t)
    if isinstance(t, Literal):
        return str(t.toPython())
    return str(t)


def load_triples_from_ttl(ttl_path: str | Path) -> TripleList:
    """
    Load triples from a Turtle (.ttl) file into a TripleList of Triple objects.
    """
    path = Path(ttl_path)
    g = RDFGraph()
    g.parse(str(path), format="turtle")

    out: TripleList = []
    for s, p, o in g:
        out.append(Triple(s=_term_to_str(s), p=_term_to_str(p), o=_term_to_str(o)))

    return out


def write_triples_to_nt(triples: Iterable[Triple], out_path: str | Path) -> None:
    """
    Optional helper: write triples as N-Triples (very general).
    This avoids needing namespace config. Useful for quick debug exports.
    """
    path = Path(out_path)
    g = RDFGraph()
    for s, p, o in triples:
        g.add(
            (
                URIRef(s),
                URIRef(p),
                Literal(o) if not str(o).startswith("http") else URIRef(o),
            )
        )
    g.serialize(destination=str(path), format="nt")
