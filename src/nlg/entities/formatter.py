from __future__ import annotations
from typing import Optional, Protocol, Any

from src.nlg.grammar.articles import add_article


class EntityInfoLike(Protocol):
    name: str
    is_countable: bool


class EntityRegistryLike(Protocol):
    def get(self, eid: str) -> Optional[EntityInfoLike]: ...


class EntityFormatter:
    """
    Turns an entity id into a human-friendly phrase.
    """

    def __init__(self, registry: EntityRegistryLike, *, entity_lexicon=None, definite: bool = False):
        self.registry = registry
        self.entity_lexicon = entity_lexicon
        self.definite = definite

    def format_entity(self, eid: str) -> str:
        # lexicon override first
        if self.entity_lexicon:
            lex = self.entity_lexicon.get(eid)
            if lex:
                # treat lex.label as name for article purposes
                info = self.registry.get(eid)
                if info:
                    return add_article(lex.label, info, definite=self.definite)
                return lex.label

        # fallback to registry
        info = self.registry.get(eid)
        if not info:
            return eid
        return add_article(info.name, info, definite=self.definite)

    def format_term(self, term: Any) -> str:
        """
        If term is a string, assume it might be an entity id and try registry.
        Otherwise render literal as string.
        """
        if isinstance(term, str):
            return self.format_entity(term)
        return str(term)
