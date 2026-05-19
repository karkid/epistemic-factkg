"""Base interface for synthetic text generation clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvidenceSpec:
    """Epistemic + generation parameters for one evidence item."""

    stance: str  # EvidenceStance value
    source_id: str  # registry key → source trust
    evidence_types: list[str]
    inference_strength: float
    reliability: str  # "strong" | "weak" | "hedged"


class SyntheticTextClient(ABC):
    """Generates fictional claim + evidence texts from a list of EvidenceSpecs.

    Implementors must produce text that is semantically consistent:
    every evidence item must relate to the same claim, and the reliability
    level must be reflected in the language (hedging, directness, etc.).
    """

    @abstractmethod
    def generate(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        """Return {"claim": str, "evidence_texts": [str, ...]} or None on failure."""
