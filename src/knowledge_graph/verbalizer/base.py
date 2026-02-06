"""
Fresh ontology implementation - flexible and extensible.
"""

from typing import Dict
from enum import Enum

from knowledge_graph.verbalizer.kg_template import KGTemplate

from utils.typing import Triple

class Verbalizer:

    def __init__(self):

        self._map = Dict[str, KGTemplate]()

    def register_template(self, template: KGTemplate) -> None:
        """
        Register KGTemplate for predicate.
        """

        self._map[template.predicate] = template

    def hasTemplate(self, predicate: str) -> bool:
        """
        Check if template exists for predicate.
        """

        return predicate in self._map

    def verbalize(self, triple: Triple) -> str | None:

        tpl = self._map.get(triple.p)

        if tpl is None:
            return None

        return tpl.verbalize(triple)

