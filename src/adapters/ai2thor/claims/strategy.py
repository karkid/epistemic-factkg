"""Shared strategy/mapping helpers for AI2THOR claim generation and conversion.

Imported by both claims/types.py (generator output) and converter.py (legacy conversion)
to avoid circular imports and keep the mapping logic in one place.
"""

from __future__ import annotations

from src.epistemic.enums import EvidenceStance, EvidenceType, ReasoningStrategy, Verdict

_SPATIAL_PREDS = {"inside", "ontopof", "near", "in", "on"}
_AFFORDANCE_PREDS = {"breakable", "pickupable", "openable", "istoggleable"}

_STRATEGY_MAP: dict[str, str] = {
    "direct_observation": ReasoningStrategy.DIRECT_OBSERVATION,
    "absence_detection": ReasoningStrategy.ABSENCE_DETECTION,
    "spatial_reasoning": ReasoningStrategy.SPATIAL_COMPARISON,
    "action_testing": ReasoningStrategy.MULTI_HOP_INFERENCE,
}

_STRATEGY_EVIDENCE_TYPES: dict[str, list[str]] = {
    "direct_observation": [EvidenceType.PERCEPTION.value],
    "absence_detection": [
        EvidenceType.PERCEPTION.value,
        EvidenceType.NON_APPREHENSION.value,
    ],
    "spatial_reasoning": [
        EvidenceType.PERCEPTION.value,
        EvidenceType.COMPARISON_ANALOGY.value,
    ],
    "action_testing": [EvidenceType.PERCEPTION.value, EvidenceType.INFERENCE.value],
}


def _to_strategy(raw: str | None) -> str:
    return _STRATEGY_MAP.get(raw or "", ReasoningStrategy.DIRECT_OBSERVATION)


def _classify_strategy(predicate: str, ev_triples: list) -> str:
    pred = predicate.lower()
    if not ev_triples:
        return "absence_detection"
    if pred in _SPATIAL_PREDS:
        return "spatial_reasoning"
    if pred in _AFFORDANCE_PREDS:
        return "action_testing"
    return "direct_observation"


def _infer_evidence_types(strategy: str | None, has_ev_triples: bool) -> list[str]:
    """Return evidence_types for a single AI2THOR evidence item based on strategy.

    Strategy mapping:
      direct_observation  → [perception]
      absence_detection   → [perception, non_apprehension]
      spatial_reasoning   → [perception, comparison_analogy]
      action_testing      → [perception, inference]

    Fallback: has triples → [perception]; no triples → [perception, non_apprehension].
    """
    if strategy in _STRATEGY_EVIDENCE_TYPES:
        return _STRATEGY_EVIDENCE_TYPES[strategy]
    return (
        [EvidenceType.PERCEPTION.value]
        if has_ev_triples
        else [EvidenceType.PERCEPTION.value, EvidenceType.NON_APPREHENSION.value]
    )


def _label_to_stance(label: Verdict | None) -> str | None:
    """Derive evidence stance from verdict.

    Stance mirrors the verdict directly — absence claims included.
    """
    if label == Verdict.SUPPORTED:
        return EvidenceStance.SUPPORTS.value
    if label == Verdict.REFUTED:
        return EvidenceStance.REFUTES.value
    return None
