from enum import StrEnum

class OutputLabels(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"

class ReasoningLabels(StrEnum):
    ONE_HOP = "one-hop"
    MULTI_HOP = "multi-hop"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"

class SourceTypesLabels(StrEnum):
    PERCEPTION = "perception"
    MEMORY = "memory"
    INFERENCE = "inference"
    TESTIMONY = "testimony"
    UNKNOWN = "unknown"
