from __future__ import annotations
from .engine import get_engine


def verb_agreement(verb: str, singular: bool, tense: str = "present") -> str:
    """
    Conjugate verb with agreement using inflect.

    - present: "sit" vs "sits"
    - past: "sat"
    - be: special-case for is/are/was/were
    """
    verb = verb.strip()
    if not verb:
        return verb

    eng = get_engine()

    if tense == "past":
        if verb == "be":
            return "was" if singular else "were"
        return eng.past(verb)

    # present
    if verb == "be":
        return "is" if singular else "are"

    if not singular:
        return verb  # base form

    # 3rd person singular
    return eng.present(verb, 3, "singular")


def negate_verb(
    verb: str,
    singular: bool,
    tense: str = "present",
    contracted: bool = False,
) -> str:
    """
    Return negated verb phrase.

    Examples:
      sit + singular → does not sit
      sit + plural   → do not sit
      be + singular  → is not
      be + plural    → are not
    """
    verb = verb.strip()
    if not verb:
        return verb

    eng = get_engine()

    # BE special case
    if verb == "be":
        if tense == "present":
            base = "is" if singular else "are"
        elif tense == "past":
            base = "was" if singular else "were"
        else:
            raise ValueError("Unsupported tense")

        return f"{base}n't" if contracted else f"{base} not"

    # Other verbs
    if tense == "present":
        aux = "does" if singular else "do"
        aux = f"{aux}n't" if contracted else f"{aux} not"
        base = eng.present(verb, 1)  # base form
        return f"{aux} {base}"

    if tense == "past":
        aux = "did"
        aux = f"{aux}n't" if contracted else f"{aux} not"
        base = eng.present(verb, 1)
        return f"{aux} {base}"

    raise ValueError("Unsupported tense")
