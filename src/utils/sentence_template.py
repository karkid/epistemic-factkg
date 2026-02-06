from typing import Set
import string


import re

class SentenceNormalizer:
    """
    Post-process generated sentences:
    - Fix capitalization
    - Normalize spacing
    - Handle negation
    """

    @staticmethod
    def normalize(text: str) -> str:

        if not text:
            return text

        text = text.strip()

        # Remove double spaces
        text = re.sub(r"\s+", " ", text)

        # Capitalize first character
        text = text[0].upper() + text[1:]

        # Ensure period at end
        if not text.endswith("."):
            text += "."

        return text

class SentenceTemplate:
    """
    Template-based sentence generator.
    Example template: "{subject} is located in {object}"
    T = SentenceTemplate(
        template="The {s} is on the {o}.",
        fields={"s", "o"},
    )

    print(T.verbalize(s="mug", o="table"))
    """

    def __init__(self, template: str, fields: Set[str]):

        self.template = template
        self.fields = fields

        self._validate_template()

    # --------------------------------------------------

    def _validate_template(self) -> None:
        """
        Ensure template contains required placeholders.
        """

        found_fields = self._parse_fields()

        missing = self.fields - found_fields

        if missing:
            raise ValueError(
                f"Missing fields in template: {missing}"
            )

    # --------------------------------------------------

    def _parse_fields(self) -> Set[str]:
        """
        Extract format field names from template.
        """

        formatter = string.Formatter()

        fields = set()

        for _, field_name, _, _ in formatter.parse(self.template):

            if field_name is not None:
                fields.add(field_name)

        return fields

    # --------------------------------------------------

    def verbalize(self, **kwargs) -> str:
        """
        Fill the template with values.
        """

        missing = self.fields - kwargs.keys()

        if missing:
            raise ValueError(
                f"Missing values for: {missing}"
            )

        return self.template.format(**kwargs)
