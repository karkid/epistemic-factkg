from __future__ import annotations

from src.nlg.lexicon.predicates import PredicateLexicon, PredicateLexeme
from src.nlg.lexicon.entities import EntityLexicon, EntityLexeme


def build_ai2thor_predicate_lexicon() -> PredicateLexicon:
    lex = PredicateLexicon()

    # Spatial
    lex.register("onTopOf", PredicateLexeme(kind="prep", text="on"))
    lex.register("inside", PredicateLexeme(kind="prep", text="inside"))
    lex.register("near", PredicateLexeme(kind="prep", text="near"))
    lex.register("hanging", PredicateLexeme(kind="prep", text="hanging on"))

    # State
    lex.register("isOpen", PredicateLexeme(kind="adj", text="open"))
    lex.register("isDirty", PredicateLexeme(kind="adj", text="dirty"))
    lex.register("isBroken", PredicateLexeme(kind="adj", text="broken"))
    lex.register("isToggled", PredicateLexeme(kind="adj", text="switched on"))

    # Data
    lex.register("hasTemperature", PredicateLexeme(kind="attr", text="temperature"))
    lex.register("hasMass", PredicateLexeme(kind="attr", text="mass"))
    lex.register("hasMaterial", PredicateLexeme(kind="attr", text="material"))

    return lex


def build_ai2thor_entity_lexicon() -> EntityLexicon:
    lex = EntityLexicon()

    # Only put exceptions here. Everything else can fall back to eid or EntityInfo.name.
    lex.register("TVStand", EntityLexeme(label="TV stand"))
    lex.register("PaperTowelRoll", EntityLexeme(label="paper towel roll"))
    lex.register("ArmChair", EntityLexeme(label="armchair"))

    return lex
