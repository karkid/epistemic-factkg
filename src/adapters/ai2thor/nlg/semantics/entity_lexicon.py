from __future__ import annotations

import re

from src.ontology.semantics.entity_lexicon import EntityLexeme, EntityLexicon

from src.adapters.ai2thor.knowledge.ids.object_types import (
    PICKABLE_OBJECTS,
    ObjectType,
    ACRONYMS,
    COUNTABLE_OBJECTS,
)


# ---- helpers ----


# Split only when: lowercase/digit → Uppercase
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _split_camel(s: str) -> str:
    """
    Convert CamelCase to space-separated words.
    Example: PaperTowelRoll -> Paper Towel Roll
             TVStand -> TV Stand
             CD -> CD
    """
    return _CAMEL_SPLIT.sub(" ", s)


def _fix_acronyms(text: str) -> str:
    """
    Normalize known acronyms without breaking them.
    """

    def replace(match: re.Match) -> str:
        word = match.group(0)

        upper = word.upper()
        if upper in ACRONYMS:
            return upper

        return word

    # Replace only full words
    return re.sub(r"\b\w+\b", replace, text)


def _normalize_case(text: str) -> str:
    words = []

    for w in text.split():
        if w.upper() in ACRONYMS:
            words.append(w.upper())
        else:
            words.append(w.lower())

    return " ".join(words)


def _fix_special_cases(text: str) -> str:
    """
    Handle any special cases that aren't covered by the general rules.
    Example: "TV Stand" -> "tv stand" (after fixing acronyms, we just need to lowercase)
    """
    if text in {"Room Temp"}:
        return "at room temperature"

    # For now, we don't have any special cases, but you can add them here if needed.
    return text


def default_object_type_label(object_type: str) -> str:
    """
    Convert AI2-THOR ObjectType (CamelCase) to a human label.

    Example:
        PaperTowelRoll -> paper towel roll
        TVStand -> tv stand
        CD -> cd
    """

    # Step 1: Split CamelCase
    text = _split_camel(object_type)

    # Step 2: Fix acronyms
    text = _fix_acronyms(text)

    # Step 3: Spcial cases (if needed)
    text = _fix_special_cases(text)

    # Step 4: Normalize casing
    text = _normalize_case(text)

    return text


# ---- lexicon ----


def create_ai2thor_object_type_lexicon() -> EntityLexicon:
    """
    Create an EntityLexeme for an AI2-THOR ObjectType, using a default label.
    """

    lexicon = EntityLexicon()

    for ot in ObjectType:
        lexicon.register(
            ot,
            EntityLexeme(
                label=default_object_type_label(ot),
                is_countable=ot in COUNTABLE_OBJECTS,
                proper=False,
                mass_noun=ot in PICKABLE_OBJECTS
                and ot
                not in COUNTABLE_OBJECTS,  # e.g. "water" is a mass noun, but "knife" isn't
            ),
        )

    return lexicon
