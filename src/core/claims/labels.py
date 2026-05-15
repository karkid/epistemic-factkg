from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse


class Verdict(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    ABSENT = "absent"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class ClaimStructure(StrEnum):
    ONE_HOP = "one_hop"
    MULTI_HOP = "multi_hop"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"
    ABSENCE = "absence"


class EvidenceType(StrEnum):
    """Per-evidence epistemic categories (Pramana-derived). Used at evidence level, multi-label."""
    PERCEPTION = "perception"
    NON_APPREHENSION = "non_apprehension"
    TESTIMONY = "testimony"
    COMPARISON_ANALOGY = "comparison_analogy"
    INFERENCE = "inference"
    POSTULATION_DERIVATION = "postulation_derivation"


# Backward-compatible alias — converters still referencing Pramana continue to work
# until they are updated in Steps 4 and 5.
Pramana = EvidenceType


class ReasoningStrategy(StrEnum):
    """Unified reasoning strategy taxonomy across all three sources.

    Assigned at the claim level — describes HOW the claim is verified.
    Used as a 6-d one-hot on claim nodes in the GNN (not on evidence nodes).
    """
    DIRECT_OBSERVATION   = "direct_observation"   # AI2THOR: sensor reads property directly
    ABSENCE_DETECTION    = "absence_detection"    # AI2THOR: sensor confirms object absent
    SPATIAL_COMPARISON   = "spatial_comparison"   # AI2THOR: spatial relation; AVeriTeC: numeric
    TESTIMONIAL_LOOKUP   = "testimonial_lookup"   # AVeriTeC/synthetic: written evidence lookup
    MULTI_HOP_INFERENCE  = "multi_hop_inference"  # AI2THOR action_testing; AVeriTeC consultation
    CONFLICTING_EVIDENCE = "conflicting_evidence" # Synthetic: opposing evidence templates



# EW_i weights used in EC_i = 1 - (1 - ST_i)^(EW_i * IS_i).
# These are the epistemic-type weights for the diminishing-returns formula.
CONFIDENCE_WEIGHTS: dict[EvidenceType, float] = {
    EvidenceType.PERCEPTION: 0.95,
    EvidenceType.TESTIMONY: 0.80,
    EvidenceType.NON_APPREHENSION: 0.75,
    EvidenceType.COMPARISON_ANALOGY: 0.65,
    EvidenceType.INFERENCE: 0.55,
    EvidenceType.POSTULATION_DERIVATION: 0.40,
}

TRAINING_EVIDENCE_TYPES: frozenset[str] = frozenset(
    {
        EvidenceType.PERCEPTION,
        EvidenceType.TESTIMONY,
        EvidenceType.NON_APPREHENSION,
        EvidenceType.COMPARISON_ANALOGY,
        EvidenceType.INFERENCE,
    }
)

# Backward-compatible alias
TRAINING_PRAMANA = TRAINING_EVIDENCE_TYPES

# EC_i below this floor is overridden to stance = not_enough_evidence
MIN_EVIDENCE_CONFIDENCE: float = 0.10

# Fallback ST when source_id is not in registry
DEFAULT_SOURCE_TRUST: float = 0.40

# Verdict derivation thresholds (see ADR-014)
SUPPORT_THRESHOLD: float = 0.75
REFUTE_THRESHOLD: float = 0.75
CONFLICT_FLOOR: float = 0.40

# Known social-media domains for TLD heuristic fallback
_SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset(
    {
        "twitter.com", "x.com", "facebook.com", "instagram.com",
        "reddit.com", "tiktok.com", "linkedin.com", "youtube.com",
        "t.co", "fb.com",
    }
)

# Compound TLDs that must be preserved as a unit (e.g. bbc.co.uk → bbc, not bbc_co)
_COMPOUND_TLDS: frozenset[str] = frozenset({
    "co.uk", "gov.uk", "ac.uk", "org.uk", "me.uk", "net.uk",
    "co.in", "gov.in", "ac.in",
    "co.za", "gov.za", "ac.za",
    "gov.au", "com.au", "edu.au", "net.au", "org.au",
    "gov.ng", "gov.gh", "gov.ke",
    "gouv.fr",
})


# ---------------------------------------------------------------------------
# EW_i computation
# ---------------------------------------------------------------------------

def combine_evidence_weights(evidence_types: list[str], weights: dict | None = None) -> float:
    """Diminishing-returns combination: 1 - Π(1 - wᵢ).

    Computes EW_i for an evidence item with multiple epistemic types.
    A single type returns its own weight; multiple types always yield a higher
    combined weight than any individual, with the strongest dominating.
    """
    w = weights or CONFIDENCE_WEIGHTS
    complement = 1.0
    for et in evidence_types:
        complement *= 1.0 - w.get(et, 0.0)
    return round(1.0 - complement, 4)


# Backward-compatible alias — remove once all adapters use combine_evidence_weights
combine_pramana_weights = combine_evidence_weights


# ---------------------------------------------------------------------------
# Per-evidence confidence (EC_i)
# ---------------------------------------------------------------------------

def compute_evidence_confidence(st: float, ew: float, is_: float) -> float:
    """EC_i = 1 - (1 - ST_i)^(EW_i * IS_i).

    Args:
        st:  Source trustworthiness ST_i from registry (0–1).
        ew:  Epistemic-type weight EW_i = combine_evidence_weights(evidence_types) (0–1).
        is_: Inference strength IS_i from rubric (0–1).

    Returns:
        EC_i in [0, 1] rounded to 4 decimal places.
    """
    exponent = ew * is_
    if exponent == 0.0:
        return 0.0
    return round(1.0 - (1.0 - st) ** exponent, 4)


# ---------------------------------------------------------------------------
# Verdict derivation from evidence aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(evidence_items: list[dict], registry: dict[str, dict] | None = None) -> tuple[float, float]:
    """Compute (support_score, refute_score) from a list of evidence item dicts.

    Evidence items with stance 'not_enough_evidence' or 'conflicting_evidence'
    are excluded from both aggregations. 'absent' stance counts as supporting
    for non_apprehension claims (sensor-confirmed absence verifies the claim).

    Args:
        evidence_items: List of evidence dicts (schema v3.0).
        registry:       Source trust registry dict {source_id: record}.
                        If None, DEFAULT_SOURCE_TRUST is used for all items.

    Returns:
        (support_score, refute_score) — both in [0, 1].
    """
    reg = registry or {}
    support_complements: list[float] = []
    refute_complements: list[float] = []

    for ev in evidence_items:
        stance = ev.get("stance", "")
        if stance in (EvidenceStance.NOT_ENOUGH_EVIDENCE, EvidenceStance.CONFLICTING_EVIDENCE):
            continue

        ec = _compute_ec_for_item(ev, reg)

        if stance in (EvidenceStance.SUPPORTS, EvidenceStance.ABSENT):
            support_complements.append(1.0 - ec)
        elif stance == EvidenceStance.REFUTES:
            refute_complements.append(1.0 - ec)

    support_score = _product_complement(support_complements)
    refute_score = _product_complement(refute_complements)
    return round(support_score, 4), round(refute_score, 4)


def derive_verdict(support_score: float, refute_score: float) -> str:
    """Derive verdict label from aggregated support and refute scores.

    Thresholds (ADR-014):
      supported            : support >= 0.75 AND refute < 0.40
      refuted              : refute  >= 0.75 AND support < 0.40
      conflicting_evidence : support >= 0.40 AND refute >= 0.40
      not_enough_evidence  : everything else
    """
    if support_score >= SUPPORT_THRESHOLD and refute_score < CONFLICT_FLOOR:
        return Verdict.SUPPORTED
    if refute_score >= REFUTE_THRESHOLD and support_score < CONFLICT_FLOOR:
        return Verdict.REFUTED
    if support_score >= CONFLICT_FLOOR and refute_score >= CONFLICT_FLOOR:
        return Verdict.CONFLICTING_EVIDENCE
    return Verdict.NOT_ENOUGH_EVIDENCE


# ---------------------------------------------------------------------------
# Training record filter
# ---------------------------------------------------------------------------

def is_training_record(record: dict) -> bool:
    """True if the record contains at least one training EvidenceType (ADR-011).

    In schema v3.0, checks evidence_types_all in the epistemic block.
    Records whose every evidence type is postulation_derivation are excluded.
    """
    evidence_types_all = record.get("epistemic", {}).get("evidence_types_all", [])
    return bool(set(evidence_types_all) & TRAINING_EVIDENCE_TYPES)


# ---------------------------------------------------------------------------
# Source trust registry
# ---------------------------------------------------------------------------

def load_source_trust_registry(path: str | Path) -> dict[str, dict]:
    """Load source_trust_registry.jsonl into a dict keyed by source_id.

    Returns:
        {source_id: full registry record dict}
    """
    registry: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                registry[record["source_id"]] = record
    return registry


def make_source_id(domain: str, modality: str) -> str:
    """Build the candidate registry key for a domain + modality pair (no registry needed).

    Strips TLD so make_source_id("reuters.com", "web_text") → "reuters_web_text".
    Use resolve_source_id for full fallback resolution including TLD heuristics.
    """
    domain = domain.lower().removeprefix("www.")
    return f"{_normalise_domain(domain)}_{modality}"


def resolve_source_id(domain: str, modality: str, registry: dict[str, dict]) -> str:
    """Resolve a domain + modality pair to a registry source_id.

    Lookup order:
      1. Exact match:      {domain}_{modality}
      2. Parent domain:    strip first subdomain, retry exact
      3. Archive extract:  web.archive.org — parse embedded original URL, recurse
      4. TLD heuristic:    *.gov, *.edu, *.org
      5. Social media:     known social-media domains
      6. Modality default: web_table → general_web_table, pdf → academic_pdf
      7. Final fallback:   unknown_web
    """
    domain = domain.lower().removeprefix("www.")

    # 1. Exact match
    key = f"{_normalise_domain(domain)}_{modality}"
    if key in registry:
        return key

    # 2. Parent domain (strip first subdomain component)
    parent = _strip_subdomain(domain)
    if parent != domain:
        key = f"{_normalise_domain(parent)}_{modality}"
        if key in registry:
            return key

    # 3. Archive: original URL is embedded in web.archive.org paths
    if domain == "web.archive.org":
        return "webarchive_web_text"  # converter must resolve original separately

    # 4. TLD heuristics
    tld_key = _tld_heuristic(domain, modality)
    if tld_key and tld_key in registry:
        return tld_key

    # 5. Known social media
    if domain in _SOCIAL_MEDIA_DOMAINS or _strip_subdomain(domain) in _SOCIAL_MEDIA_DOMAINS:
        return "social_media_web_text"

    # 6. Modality default
    if modality == "web_table":
        return "general_web_table"
    if modality == "pdf":
        return "academic_pdf"

    # 7. Final fallback
    return "unknown_web"


def get_source_trust(source_id: str, registry: dict[str, dict]) -> float:
    """Return ST_i for a source_id, or DEFAULT_SOURCE_TRUST if not found."""
    entry = registry.get(source_id)
    return entry["source_trust"] if entry else DEFAULT_SOURCE_TRUST


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_ec_for_item(ev: dict, registry: dict[str, dict]) -> float:
    source_id = ev.get("source_id", "unknown_web")
    st = get_source_trust(source_id, registry)
    ew = combine_evidence_weights(ev.get("evidence_types", []))
    is_ = ev.get("inference_strength", 0.6)
    ec = compute_evidence_confidence(st, ew, is_)
    return max(ec, 0.0)


def _product_complement(complements: list[float]) -> float:
    if not complements:
        return 0.0
    result = 1.0
    for c in complements:
        result *= c
    return 1.0 - result


def _normalise_domain(domain: str) -> str:
    """Strip TLD and convert to snake_case for registry key lookup.

    reuters.com → reuters, wikipedia.org → wikipedia,
    bbc.co.uk → bbc (compound TLD stripped), indiatoday.in → indiatoday.
    """
    parts = domain.split(".")
    if len(parts) >= 3:
        compound = ".".join(parts[-2:])
        if compound in _COMPOUND_TLDS:
            # Strip both TLD components (e.g. ".co.uk") → keep brand only
            name = ".".join(parts[:-2])
            return name.replace(".", "_").replace("-", "_")
    if len(parts) >= 2:
        # Strip single TLD: reuters.com → reuters
        name = ".".join(parts[:-1])
        return name.replace(".", "_").replace("-", "_")
    return domain.replace(".", "_").replace("-", "_")


def _strip_subdomain(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) > 2:
        return ".".join(parts[1:])
    return domain


def _tld_heuristic(domain: str, modality: str) -> str | None:
    """Return a registry source_id based on domain TLD pattern.

    Maps to the actual registry keys: government_web_text, government_pdf,
    academic_pdf, general_web_text.  Returns None for .org (too heterogeneous).
    """
    gov_suffixes = (".gov", ".gov.uk", ".gov.au", ".gov.in", ".gov.za",
                    ".gov.ng", ".gov.gh", ".gov.ke", ".gouv.fr")
    edu_suffixes = (".edu", ".ac.uk", ".ac.in", ".ac.za", ".edu.au")

    for s in gov_suffixes:
        if domain.endswith(s):
            return "government_pdf" if modality == "pdf" else "government_web_text"
    for s in edu_suffixes:
        if domain.endswith(s):
            return "academic_pdf" if modality == "pdf" else "general_web_text"
    return None
