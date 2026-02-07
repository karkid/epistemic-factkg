# src/nlg/templates/default_templates.py
from __future__ import annotations

from typing import Dict

from src.nlg.templates.sentence_template import SentenceTemplate


def build_default_templates() -> Dict[str, SentenceTemplate]:
    """
    Returns a predicate-uri (or predicate-name) -> SentenceTemplate mapping.

    You can keep keys as:
    - full URI (recommended if your triples use URIs)
    - or short predicate strings (if your triples use short names)
    """
    return {
        "onTopOf": SentenceTemplate(
            template="{s} is {prep} {o}",
            fields={"s", "prep", "o"},
        ),
        "in": SentenceTemplate(
            template="{s} is {prep} {o}",
            fields={"s", "prep", "o"},
        ),
        "under": SentenceTemplate(
            template="{s} is {prep} {o}",
            fields={"s", "prep", "o"},
        ),
        "hasAttribute": SentenceTemplate(
            template="{s} has attribute {o}",
            fields={"s", "o"},
        ),
    }
