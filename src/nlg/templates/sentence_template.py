# src/nlg/templates/sentence_template.py
from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Set, Dict, Any


class SentenceNormalizer:
    """
    Post-process generated sentences:
    - Normalize spacing
    - Fix capitalization
    - Ensure terminal punctuation
    """

    _space_re = re.compile(r"\s+")

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return text

        text = text.strip()

        # collapse whitespace
        text = SentenceNormalizer._space_re.sub(" ", text)

        # capitalize first character
        if text:
            text = text[0].upper() + text[1:]

        # ensure terminal punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text


class SentenceTemplate:
    """
    Template-based sentence generator.

    Example:
        t = SentenceTemplate(
            template="{s} is on {o}",
            fields={"s", "o"},
            normalize=True,
        )
        t.verbalize(s="a mug", o="the table")  # -> "A mug is on the table."
    """

    def __init__(self, template: str, fields: Set[str], *, normalize: bool = True):
        self.template = template
        self.fields = set(fields)
        self.normalize = normalize
        self._validate_template()

    def _parse_fields(self) -> Set[str]:
        """Extract format field names from template."""
        formatter = string.Formatter()
        found: Set[str] = set()
        for _, field_name, _, _ in formatter.parse(self.template):
            if field_name:  # skips None and ""
                found.add(field_name)
        return found

    def _validate_template(self) -> None:
        """Ensure template contains required placeholders."""
        found_fields = self._parse_fields()
        missing = self.fields - found_fields
        if missing:
            raise ValueError(f"Missing fields in template: {sorted(missing)}")

    def verbalize(self, **kwargs: Any) -> str:
        """Fill the template with values."""
        missing = self.fields - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing values for: {sorted(missing)}")

        text = self.template.format(**kwargs)
        return SentenceNormalizer.normalize(text) if self.normalize else text
