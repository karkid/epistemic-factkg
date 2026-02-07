from __future__ import annotations
from .engine import get_engine


def pluralize(noun: str, count: int) -> str:
    """
    Return correct plural form based on count.
    """
    noun = noun.strip()
    if not noun:
        return noun

    if count == 1:
        return noun

    eng = get_engine()
    return eng.plural(noun, count)
