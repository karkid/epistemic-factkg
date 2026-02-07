from __future__ import annotations

import random
from dataclasses import dataclass
from typing import FrozenSet, Optional
from urllib.parse import unquote

from src.core.graph.types import Triple, TripleList, TripleSet
from src.semantics.claims.base import BaseClaimCorruptor


BOOL_PREDICATES: FrozenSet[str] = frozenset({"isOpen", "isToggled", "isDirty", "isBroken"})

AI2THOR_CONTAINER_OBJECT_TYPES: FrozenSet[str] = frozenset(
    {"Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "Toilet", "Box", "SinkBasin", "BathtubBasin"}
)

AI2THOR_SURFACE_OBJECT_TYPES: FrozenSet[str] = frozenset(
    {"StoveBurner", "CounterTop", "DiningTable", "CoffeeTable", "SideTable", "Desk", "Dresser",
     "TVStand", "Shelf", "Sofa", "ArmChair", "Ottoman", "Bed"}
)


def _short(uri: str) -> str:
    u = unquote(uri)
    u = u.split("#")[-1]
    u = u.rsplit("/", 1)[-1]
    return u


def _is_type_predicate(pred: str) -> bool:
    return _short(pred).lower() in {"type", "rdf:type"}


def is_boolean_predicate(pred: str) -> bool:
    ps = _short(pred)
    return ps in BOOL_PREDICATES or any(ps.lower() == b.lower() for b in BOOL_PREDICATES)


def is_inside_predicate(pred: str) -> bool:
    ps = _short(pred).lower()
    return ps == "inside" or ps.endswith("inside")


def is_ontopof_predicate(pred: str) -> bool:
    ps = _short(pred).lower()
    return ps == "ontopof" or ps.endswith("ontopof")


def entity_objtype(uri: str) -> str:
    tail = _short(uri)  # URL-decoded
    if "|" in tail:
        return tail.split("|", 1)[0]
    if "_" in tail:
        return tail.split("_", 1)[0]
    return tail


def flip_bool_object(o: object) -> str:
    ol = str(o).strip().lower()
    if ol in {"true", "1", "yes"}:
        return "False"
    if ol in {"false", "0", "no"}:
        return "True"
    return "True"


@dataclass
class AI2ThorClaimCorruptor(BaseClaimCorruptor):
    """
    AI2-THOR-specific corruptor:
    - boolean: flip True/False
    - inside: swap object to a different container entity
    - onTopOf: swap object to a different surface entity
    - fallback: swap object to any other existing entity
    - final: swap predicate (excluding rdf:type)
    """

    seed: Optional[int] = None
    max_object_trials: int = 200
    max_predicate_trials: int = 100

    def __post_init__(self) -> None:
        if self.seed is not None:
            random.seed(self.seed)

    def corrupt(self, triple: Triple, all_triples: TripleList, triple_set: TripleSet) -> Triple:
        s, p, o = triple

        # 1) boolean flip
        if is_boolean_predicate(p):
            bad = Triple(s, p, flip_bool_object(o))
            if bad not in triple_set:
                return bad

        # Candidate pool: only existing objects from the KG
        objects = list({t.o for t in all_triples if t.o != o})

        # 2) predicate-aware object swap
        if is_inside_predicate(p):
            containers = [x for x in objects if entity_objtype(str(x)) in AI2THOR_CONTAINER_OBJECT_TYPES]
            random.shuffle(containers)
            for cand in containers[: self.max_object_trials]:
                bad = Triple(s, p, cand)
                if bad not in triple_set:
                    return bad

        if is_ontopof_predicate(p):
            surfaces = [x for x in objects if entity_objtype(str(x)) in AI2THOR_SURFACE_OBJECT_TYPES]
            random.shuffle(surfaces)
            for cand in surfaces[: self.max_object_trials]:
                bad = Triple(s, p, cand)
                if bad not in triple_set:
                    return bad

        # 3) general object swap fallback
        random.shuffle(objects)
        for cand in objects[: self.max_object_trials]:
            bad = Triple(s, p, cand)
            if bad not in triple_set:
                return bad

        # 4) predicate swap fallback (exclude rdf:type)
        predicates = list({t.p for t in all_triples if t.p != p and not _is_type_predicate(t.p)})
        random.shuffle(predicates)
        for candp in predicates[: self.max_predicate_trials]:
            bad = Triple(s, candp, o)
            if bad not in triple_set:
                return bad

        return triple
