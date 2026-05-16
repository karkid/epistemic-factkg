"""Abstract base for adapter strategy mappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StrategyMapper(ABC):
    """Maps source-specific strategy values to canonical ReasoningStrategy strings."""

    @abstractmethod
    def map_strategy(self, raw_value: Any) -> str:
        """Map source-specific strategy value to canonical ReasoningStrategy."""
