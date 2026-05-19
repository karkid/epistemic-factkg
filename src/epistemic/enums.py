"""
Epistemic enums — the canonical vocabulary for fact verification.
"""

from enum import StrEnum


class Verdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class ClaimStructure(StrEnum):
    ONE_HOP = "one_hop"
    MULTI_HOP = "multi_hop"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"
    ABSENCE = "absence"


class EvidenceType(StrEnum):
    """Per-evidence epistemic categories (EvidenceType-derived). Used at evidence level, multi-label."""

    PERCEPTION = "perception"
    NON_APPREHENSION = "non_apprehension"
    TESTIMONY = "testimony"
    COMPARISON_ANALOGY = "comparison_analogy"
    INFERENCE = "inference"
    POSTULATION_DERIVATION = "postulation_derivation"


class ReasoningStrategy(StrEnum):
    """Unified reasoning strategy taxonomy across all three sources.

    Assigned at the claim level — describes HOW the claim is verified.
    Used as a 6-d one-hot on claim nodes in the GNN (not on evidence nodes).
    """

    DIRECT_OBSERVATION = "direct_observation"  # AI2THOR: sensor reads property directly
    ABSENCE_DETECTION = "absence_detection"  # AI2THOR: sensor confirms object absent
    SPATIAL_COMPARISON = (
        "spatial_comparison"  # AI2THOR: spatial relation; AVeriTeC: numeric
    )
    TESTIMONIAL_LOOKUP = (
        "testimonial_lookup"  # AVeriTeC/synthetic: written evidence lookup
    )
    MULTI_HOP_INFERENCE = (
        "multi_hop_inference"  # AI2THOR action_testing; AVeriTeC consultation
    )
    CONFLICTING_EVIDENCE = (
        "conflicting_evidence"  # Synthetic: opposing evidence templates
    )
