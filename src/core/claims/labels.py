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
    Pramana.PERCEPTION: 0.95,  # Simulator ground truth — exact
    Pramana.TESTIMONY: 0.80,  # Web sources — reliable but noisy
    Pramana.NON_APPREHENSION: 0.75,  # Absence reasoning — closed world only
    Pramana.COMPARISON_ANALOGY: 0.65,  # Analogical/statistical reasoning
    Pramana.INFERENCE: 0.55,  # Multi-step synthesis — compounds error
    Pramana.POSTULATION_DERIVATION: 0.40,  # Hypothetical — least reliable
}


TRAINING_PRAMANA: frozenset[str] = frozenset(
    {
        Pramana.PERCEPTION,
        Pramana.TESTIMONY,
        Pramana.NON_APPREHENSION,
        Pramana.COMPARISON_ANALOGY,
        Pramana.INFERENCE,
    }
)


def is_training_record(record: dict) -> bool:
    """True if record's pramana_primary is included in GNN training (ADR-011)."""
    pramana = record.get("epistemic", {}).get("pramana_primary", "")
    return pramana in TRAINING_PRAMANA


def combine_pramana_weights(pramanas: list[str], weights: dict | None = None) -> float:
    """Diminishing returns combination: 1 - Π(1 - wᵢ).

    Models 'at least one epistemic source is correct'. A single pramana returns
    its own weight. Multiple pramanas always yield a higher combined weight than
    any individual, with the strongest source dominating.
    """
    w = weights or CONFIDENCE_WEIGHTS
    complement = 1.0
    for p in pramanas:
        complement *= 1.0 - w.get(p, 0.0)
    return round(1.0 - complement, 4)
