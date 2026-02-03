from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from rdflib import Graph, URIRef, Literal

Triple = Tuple[str, str, str]
from urllib.parse import unquote
import re


@dataclass
class RDFLoadResult:
    triples: List[Triple]
    triple_set: Set[Triple]


def _term_to_str(t) -> str:
    if isinstance(t, URIRef):
        return str(t)
    if isinstance(t, Literal):
        # Normalize booleans/numbers as strings for consistency
        return str(t.toPython())
    return str(t)


def load_ttl_triples(ttl_path: str) -> RDFLoadResult:
    """
    Load a Turtle file and return triples as strings.
    """
    g = Graph()
    g.parse(ttl_path, format="turtle")

    triples: List[Triple] = []
    for s, p, o in g:
        triples.append((_term_to_str(s), _term_to_str(p), _term_to_str(o)))

    # De-duplicate while preserving order
    seen: Set[Triple] = set()
    dedup: List[Triple] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            dedup.append(t)

    return RDFLoadResult(triples=dedup, triple_set=set(dedup))



def _short(x: str) -> str:
    x = unquote(x)
    if "#" in x:
        x = x.split("#")[-1]
    if "/" in x:
        x = x.split("/")[-1]
    return x

def _extract_floorplan(uri: str) -> str | None:
    u = unquote(uri)
    m = re.search(r"(FloorPlan\d+)", u, flags=re.IGNORECASE)
    return m.group(1) if m else None

def build_object_to_scene(triples: List[Triple]) -> Dict[str, str]:
    obj2scene: Dict[str, str] = {}
    for s, p, o in triples:
        pred = _short(p).lower()

        if pred == "hasobject":
            fp = _extract_floorplan(s)
            if fp:
                obj2scene[o] = fp

        elif pred == "inscene":
            fp = _extract_floorplan(o)
            if fp:
                obj2scene[s] = fp

    return obj2scene

def group_by_floorplan(triples: List[Triple]) -> Dict[str, List[Triple]]:
    obj2scene = build_object_to_scene(triples)
    out: Dict[str, List[Triple]] = {}

    for s, p, o in triples:
        fp = _extract_floorplan(s) or _extract_floorplan(o)  # direct case

        # If not directly in this triple, use mapping
        if not fp:
            fp = obj2scene.get(s) or obj2scene.get(o) or "UNKNOWN"

        out.setdefault(fp, []).append((s, p, o))

    return out
