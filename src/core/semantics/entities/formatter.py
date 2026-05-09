from __future__ import annotations

import re
import inflect
from typing import Optional

from src.core.semantics.lexicon.entities import EntityLexicon, EntityLexeme


_engine = inflect.engine()


def _clean_label(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


class EntityFormatter:
    """
    Formats entity/object type labels with articles and pluralization.

    This formatter is intentionally type-based (object_type -> phrase).
    If later you want entity-id formatting ("Mug|..."), resolve id->type first.
    """

    def __init__(
        self, entity_lexicon: EntityLexicon, *, default_definite: bool = False
    ):

        self.lex = entity_lexicon
        self.default_definite = default_definite

    def format_type(
        self,
        object_type: str,
        *,
        definite: Optional[bool] = None,
        count: int = 1,
    ) -> str:
        """
        Examples:
          format_type("Mug") -> "a mug"
          format_type("Mug", definite=True) -> "the mug"
          format_type("Mug", count=2) -> "two mugs"
        """

        definite = self.default_definite if definite is None else definite

        lexeme = self.lex.get(object_type)
        if lexeme is None:
            base = _clean_label(object_type).lower()
            return self._apply_determiner(
                base, definite=definite, count=count, lexeme=None
            )

        base = _clean_label(lexeme.label)

        # proper nouns: no article
        if lexeme.proper:
            return base

        if lexeme.mass_noun:
            return f"the {base}" if definite else base

        # countable nouns
        if count != 1 and lexeme.is_countable:
            plural = _engine.plural_noun(base, count) or base
            return f"{count} {plural}"

        # singular countable: add article
        return self._apply_determiner(base, definite=definite, count=1, lexeme=lexeme)

    def _apply_determiner(
        self, base: str, *, definite: bool, count: int, lexeme: Optional[EntityLexeme]
    ) -> str:

        if definite:
            return f"the {base}"

        return _engine.a(base)
