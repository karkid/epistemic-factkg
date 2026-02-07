from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict


class BaseTemplate(ABC):
    @abstractmethod
    def matches(self, predicate: str) -> bool:
        """Check if this template can handle predicate."""
        raise NotImplementedError

    @abstractmethod
    def render(self, slots: Dict[str, str]) -> str:
        """Produce a sentence from slots."""
        raise NotImplementedError
