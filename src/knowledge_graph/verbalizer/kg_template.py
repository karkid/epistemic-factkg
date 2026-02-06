from typing import Set, Optional
from utils.typing import Triple
from utils.sentence_template import SentenceNormalizer, SentenceTemplate


class KGTemplate:
    """
    Template for verbalizing a specific KG predicate.
    """

    def __init__(
        self,
        *,
        predicate: str,
        positive: SentenceTemplate,
        negative: SentenceTemplate | None = None,
        normalizer: SentenceNormalizer | None = None,
    ):
        self.predicate = predicate
        self.positive = positive
        self.negative = negative
        self.normalizer = normalizer or SentenceNormalizer()

    # --------------------------------------------------

    def verbalize(
        self,
        triple: Triple,
        *,
        negated: bool = False,
    ) -> str | None:
        """
        Verbalize triple using template.
        """

        if triple.p != self.predicate:
            return None

        s = self._normalize_entity(triple.s)
        o = self._normalize_entity(triple.o)

        if negated:
            if not self.negative:
                raise ValueError(
                    f"No negative template for {self.predicate}"
                )
            text = self.negative.verbalize(s=s, o=o)
        else:
            text = self.positive.verbalize(s=s, o=o)

        return self.normalizer.normalize(text)


    # --------------------------------------------------

    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize entity string.
        """

        return entity.replace("_", " ")
