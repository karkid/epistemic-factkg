from enum import StrEnum


class Verdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    ABSENT = "absent"


class ClaimStructure(StrEnum):
    ONE_HOP = "one_hop"
    MULTI_HOP = "multi_hop"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"
    ABSENCE = "absence"


class Pramana(StrEnum):
    PERCEPTION = "perception"
    NON_APPREHENSION = "non_apprehension"
    TESTIMONY = "testimony"
    COMPARISON_ANALOGY = "comparison_analogy"
    INFERENCE = "inference"
    POSTULATION_DERIVATION = "postulation_derivation"


CONFIDENCE_WEIGHTS: dict[Pramana, float] = {
    Pramana.PERCEPTION: 0.90,
    Pramana.TESTIMONY: 0.85,
    Pramana.NON_APPREHENSION: 0.80,
    Pramana.COMPARISON_ANALOGY: 0.75,
    Pramana.INFERENCE: 0.70,
    Pramana.POSTULATION_DERIVATION: 0.60,
}
