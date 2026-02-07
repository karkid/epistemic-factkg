from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseRealizer(ABC):
    @abstractmethod
    def realize(self, triple: Any) -> str:
        """Convert KG triple to sentence."""
        raise NotImplementedError
