from enum import StrEnum


class OutputLabels(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"


class ReasoningLabels(StrEnum):
    ONE_HOP = "one-hop"
    MULTI_HOP = "multi-hop"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"


class PramanaLabel(StrEnum):
    PERCEPTION = "perception"
    NON_APPREHENSION = "non_apprehension"
    TESTIMONY = "testimony"
    COMPARISON_ANALOGY = "comparison_analogy"
    INFERENCE = "inference"
    POSTULATION_DERIVATION = "postulation_derivation"


CONFIDENCE_WEIGHTS: dict[PramanaLabel, float] = {
    PramanaLabel.PERCEPTION: 0.90,
    PramanaLabel.TESTIMONY: 0.85,
    PramanaLabel.NON_APPREHENSION: 0.80,
    PramanaLabel.COMPARISON_ANALOGY: 0.75,
    PramanaLabel.INFERENCE: 0.70,
    PramanaLabel.POSTULATION_DERIVATION: 0.60,
}


class SourceTypesLabels(StrEnum):
    PERCEPTION = "perception"
    INFERENCE = "inference"
    TESTIMONY = "testimony"
    UNKNOWN = "unknown"
    COMPARISON_ANALOGY = "comparison_analogy"
    NON_APPREHENSION = "non_apprehension"
    POSTULATION_DERIVATION = "postulation_derivation"
