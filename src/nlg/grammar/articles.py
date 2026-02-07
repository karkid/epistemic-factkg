from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

from .engine import get_engine


class HasCountability(Protocol):
    is_countable: bool


def add_indefinite_article(noun_phrase: str) -> str:
    """
    Add 'a/an' using inflect.
    """
    eng = get_engine()
    return eng.a(noun_phrase)  # returns "a ..." or "an ..."


def add_definite_article(noun_phrase: str) -> str:
    return f"the {noun_phrase}"


def add_article(noun_phrase: str, info: HasCountability, *, definite: bool = False) -> str:
    """
    If not countable (e.g., 'water'), don't add indefinite article.
    """
    noun_phrase = noun_phrase.strip()
    if not noun_phrase:
        return noun_phrase

    if definite:
        return add_definite_article(noun_phrase)

    # indefinite
    if hasattr(info, "is_countable") and not info.is_countable:
        return noun_phrase

    return add_indefinite_article(noun_phrase)
