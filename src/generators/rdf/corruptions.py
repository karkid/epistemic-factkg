from __future__ import annotations

import random
from typing import List, Set, Tuple, FrozenSet

Triple = Tuple[str, str, str]

BOOL_PRED_SUFFIXES = ("isOpen", "isToggled", "isDirty", "isBroken")

# ---- AI2-THOR type sets (string names, not enums) ----
AI2THOR_CONTAINER_OBJECT_TYPES: FrozenSet[str] = frozenset(
    [
        "Fridge",
        "Microwave",
        "Cabinet",
        "Drawer",
        "Safe",
        "Toilet",
        "Box",
        "SinkBasin",
        "BathtubBasin",
    ]
)

AI2THOR_SURFACE_OBJECT_TYPES: FrozenSet[str] = frozenset(
    [
        "StoveBurner",
        "CounterTop",
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "Desk",
        "Dresser",
        "TVStand",
        "Shelf",
        "Sofa",
        "ArmChair",
        "Ottoman",
        "Bed",
    ]
)

# ---- predicate normalization ----
def _pred_short(pred: str) -> str:
    return pred.split("#")[-1].split("/")[-1]

def is_boolean_predicate(pred: str) -> bool:
    ps = _pred_short(pred)
    return any(ps.endswith(suf) or ps.lower() == suf.lower() for suf in BOOL_PRED_SUFFIXES)

def is_inside_predicate(pred: str) -> bool:
    ps = _pred_short(pred).lower()
    return ps.endswith("inside") or ps == "inside"

def is_ontopof_predicate(pred: str) -> bool:
    ps = _pred_short(pred).lower()
    return ps.endswith("ontopof") or ps == "ontopof"

# ---- object type extraction from AI2THOR entity URI ----
def _entity_objtype(uri: str) -> str:
    """
    Example:
      http://.../entities/CoffeeMachine%7C%2B01.13%7C...
    We want: CoffeeMachine
    """
    tail = uri.rsplit("/", 1)[-1]  # CoffeeMachine%7C%2B01...
    # the obj type is before first "%7C" if present
    return tail.split("%7C", 1)[0]

def flip_bool_object(o: str) -> str:
    ol = str(o).lower()
    if ol == "true":
        return "False"
    if ol == "false":
        return "True"
    return "True"

def _is_type_predicate(pred: str) -> bool:
    """
    Detect rdf:type or any predicate ending with 'type'.
    We must exclude these because they create nonsense claims.
    """
    ps = pred.split("#")[-1].split("/")[-1].lower()
    return ps == "type"


def corrupt_triple(triple: Triple, all_triples: List[Triple], triple_set: Set[Triple]) -> Triple:
    """
    Create a likely-false triple from a true triple:
    - boolean predicates: flip object True/False
    - inside: swap object to a different *container* entity
    - onTopOf: swap object to a different *surface* entity
    - fallback: swap object to any different entity (no junk strings)
    """
    s, p, o = triple

    # 1) boolean flip
    if is_boolean_predicate(p):
        bad = (s, p, flip_bool_object(o))
        if bad not in triple_set:
            return bad

    # Build candidate pools from existing KG entities (no invented objects)
    all_entities_o = list({t[2] for t in all_triples})
    all_entities_o = [x for x in all_entities_o if x != o]

    # 2) predicate-aware pools
    if is_inside_predicate(p):
        containers = [x for x in all_entities_o if _entity_objtype(x) in AI2THOR_CONTAINER_OBJECT_TYPES]
        random.shuffle(containers)
        for cand in containers[:500]:
            bad = (s, p, cand)
            if bad not in triple_set:
                return bad

    if is_ontopof_predicate(p):
        surfaces = [x for x in all_entities_o if _entity_objtype(x) in AI2THOR_SURFACE_OBJECT_TYPES]
        random.shuffle(surfaces)
        for cand in surfaces[:500]:
            bad = (s, p, cand)
            if bad not in triple_set:
                return bad

    # 3) general object-swap fallback (still must be an existing entity)
    random.shuffle(all_entities_o)
    for cand in all_entities_o[:800]:
        bad = (s, p, cand)
        if bad not in triple_set:
            return bad

    # 4) predicate swap fallback (rare) — EXCLUDE rdf:type
    preds = list({t[1] for t in all_triples if t[1] != p and not _is_type_predicate(t[1])})
    random.shuffle(preds)
    for candp in preds[:200]:
        bad = (s, candp, o)
        if bad not in triple_set:
            return bad

    # Worst-case: return original (avoid creating non-KG garbage)
    return (s, p, o)
