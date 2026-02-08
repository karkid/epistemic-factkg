from __future__ import annotations

from src.core.semantics.lexicon.predicates import PredicateForm, PredicateLexeme, PredicateLexicon
from src.adapters.ai2thor.ids.predicates import PredicateId


def create_predicate_lexicon() -> PredicateLexicon:
    """
    AI2-THOR predicate id -> English surface form.

    Keys must match the predicate IDs emitted by the DataSource.
    """
    lex = PredicateLexicon()

    # ---------- spatial (PREP) ----------
    lex.register(PredicateId.onTopOf.value, PredicateLexeme(kind=PredicateForm.PREP, label="on"))
    lex.register(PredicateId.inside.value, PredicateLexeme(kind=PredicateForm.PREP, label="inside"))
    lex.register(PredicateId.near.value, PredicateLexeme(kind=PredicateForm.PREP, label="near"))
    lex.register(PredicateId.hanging.value, PredicateLexeme(kind=PredicateForm.PREP, label="hanging on"))

    # Core predicates for AI2-THOR
    #lex.register(PredicateId.hasObject.value, PredicateLexeme(kind=PredicateForm.VERB, label="contains"))
    #lex.register(PredicateId.inScene.value, PredicateLexeme(kind=PredicateForm.PREP, label="in"))

    # ---------- state (ADJ) ----------
    lex.register(PredicateId.isOpen.value, PredicateLexeme(kind=PredicateForm.ADJ, label="open"))
    lex.register(PredicateId.isDirty.value, PredicateLexeme(kind=PredicateForm.ADJ, label="dirty"))
    lex.register(PredicateId.isBroken.value, PredicateLexeme(kind=PredicateForm.ADJ, label="broken"))
    lex.register(PredicateId.isToggled.value, PredicateLexeme(kind=PredicateForm.ADJ, label="turned on"))
    lex.register(PredicateId.isMoving.value, PredicateLexeme(kind=PredicateForm.ADJ, label="moving"))
    lex.register(PredicateId.isPickedUp.value, PredicateLexeme(kind=PredicateForm.ADJ, label="picked up"))
    lex.register(PredicateId.isFilledWithLiquid.value, PredicateLexeme(kind=PredicateForm.ADJ, label="filled with liquid"))
    lex.register(PredicateId.isCooked.value, PredicateLexeme(kind=PredicateForm.ADJ, label="cooked"))
    lex.register(PredicateId.isSliced.value, PredicateLexeme(kind=PredicateForm.ADJ, label="sliced"))
    lex.register(PredicateId.isUsedUp.value, PredicateLexeme(kind=PredicateForm.ADJ, label="used up"))

    # ---------- value/data labels (ATTR) ----------
    lex.register(PredicateId.temperature.value, PredicateLexeme(kind=PredicateForm.PROP, label="temperature"))
    #lex.register(PredicateId.mass.value, PredicateLexeme(kind=PredicateForm.PROP, label="mass"))
    lex.register(PredicateId.material.value, PredicateLexeme(kind=PredicateForm.PROP, label="material"))

    # Optional (only if we verbalize them)
    #if hasattr(PredicateId, "position"):
        #lex.register(PredicateId.position.value, PredicateLexeme(PredicateForm.ATTR, "position"))
    #if hasattr(PredicateId, "rotation"):
        #lex.register(PredicateId.rotation.value, PredicateLexeme(PredicateForm.ATTR, "rotation"))

    # ---------- capabilities / attributes (ATTR) ----------
    # these are nicer as adjective-ish labels; your realizer decides phrasing (“is openable”, “is pickable”)
    for pid, text in [
        # (PredicateId.openable, "openable"),
        # (PredicateId.toggleable, "toggleable"),
        # (PredicateId.pickable, "pickable"),
        # (PredicateId.movable, "movable"),
        # (PredicateId.receptacle, "a receptacle"),
        # (PredicateId.cookable, "cookable"),
        # (PredicateId.sliceable, "sliceable"),
        # (PredicateId.breakable, "breakable"),
        # (PredicateId.dirtyable, "dirtyable"),
        # (PredicateId.canFillWithLiquid, "fillable with liquid"),
        # (PredicateId.canBeUsedUp, "usable up"),
    ]:
        lex.register(pid.value, PredicateLexeme(kind=PredicateForm.PROP, label=text))

    # attribute relations (VERB)

    #lex.register(PredicateId.hasAttribute.value, PredicateLexeme(kind=PredicateForm.ATTR, label="has"))
    # ---------- object relations (VERB) ----------
    # Only include if we really want to verbalize these
    #if hasattr(PredicateId, "hasObject"):
        #lex.register(PredicateId.hasObject.value, PredicateLexeme(PredicateForm.VERB, "contains"))

    return lex
