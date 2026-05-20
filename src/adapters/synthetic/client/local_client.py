"""Offline template-based synthetic text client — no API key required.

Generates varied fictional text using vocabulary pools and random entity
substitution.  Epistemic parameters are assigned by the caller; this only
provides the linguistic layer (claim + evidence sentences).
"""

from __future__ import annotations

import random
from typing import Any

from .base import EvidenceSpec, SyntheticTextClient

# ---------------------------------------------------------------------------
# Entity pools — all fictional
# ---------------------------------------------------------------------------

_BRANDS = [
    "Stellark",
    "Nexovac",
    "Trimora",
    "Crestlyn",
    "Dalvian",
    "Vorvex",
    "Lumiven",
    "Harqual",
    "Fenwick",
    "Ostara",
]
_CITIES = [
    "Northaven",
    "Crestford",
    "Marshton",
    "Quivell",
    "Valderon",
    "Ostwick",
    "Fenbury",
    "Talmoore",
    "Rivance",
    "Kelspath",
]
_INSTS = [
    "Institute of Applied Sciences",
    "Bureau of Standards",
    "Research Council",
    "Health Authority",
    "Consumer Safety Board",
    "Standards Office",
    "Analytics Division",
    "Testing Consortium",
]
_PRODUCTS = [
    "appliance",
    "beverage",
    "supplement",
    "compound",
    "treatment",
    "device",
    "formula",
    "coating",
    "filter",
    "unit",
]
_ACTIONS = [
    "contains",
    "releases",
    "reduces",
    "increases",
    "produces",
    "stores",
    "emits",
    "processes",
]
_SUBSTANCES = [
    "trace chromium",
    "residual additives",
    "microplastics",
    "polyphenols",
    "preservative compounds",
    "binding agents",
    "antioxidants",
    "mineral deposits",
    "volatile compounds",
]
_METRICS = ["42", "31", "18", "27", "55", "12", "8", "64"]
_UNITS = ["percent", "units per litre", "parts per million", "milligrams per kilogram"]

_WEAK_PREFIXES = [
    "Reportedly, ",
    "According to an unverified source, ",
    "Sources allegedly suggest that ",
    "An anonymous post claimed that ",
    "Unconfirmed reports indicate that ",
]
_HEDGED_QUALIFIERS = [
    " — though this has not been independently verified",
    " — the findings remain disputed",
    " — based on limited available data",
    " — according to unnamed sources",
]

# ---------------------------------------------------------------------------
# Per-(evidence_type, reliability) text pools
# ---------------------------------------------------------------------------

_CLAIM_POOLS: dict[str, list[str]] = {
    "testimony": [
        "{brand}'s {product} {action} elevated levels of {subst}.",
        "The {city} {inst} confirmed that {brand}'s {product} {action} {subst}.",
        "{brand} model X7 was found to {action} {subst} in standard tests.",
        "Lab results show {brand}'s {product} {action} trace amounts of {subst}.",
        "The {inst} in {city} determined that {brand}'s {product} {action} {subst}.",
    ],
    "perception": [
        "The {brand} {product} displays a green indicator light when powered on.",
        "The {city} {inst} facility window faces the east side of the building.",
        "The {brand} {product} has visible wear marks on the lower left corner.",
        "The {brand} {product} emits a faint hum during normal operation.",
        "The storage unit at the {city} {inst} has a cracked exterior panel.",
    ],
    "inference": [
        "Prolonged use of the {brand} {product} contributes to {subst} buildup.",
        "The {city} cycling lane introduction reduced vehicle travel times.",
        "The {brand} {product} efficiency decreases when ambient temperature exceeds 30°C.",
        "Increased {inst} funding in {city} correlates with lower incident rates.",
    ],
    "comparison_analogy": [
        "The {brand} {product} uses {metric} {unit} less energy than the previous model.",
        "{city} has twice as many {product} units per capita as Marshton.",
        "The {city} {inst} reported a {metric}% improvement in output this quarter.",
        "{brand}'s new {product} costs {metric}% more than the industry average.",
    ],
    "non_apprehension": [
        "No traces of {brand} {product} compounds were found in the {city} water outflow.",
        "The {city} {inst} facility has no temperature-controlled units available.",
        "There are no reported incidents associated with the {brand} {product}.",
        "No evidence of {subst} was detected in any {brand} {product} samples tested.",
    ],
}

_TEXT_POOLS: dict[str, dict[str, list[str]]] = {
    "testimony": {
        "strong": [
            "The {city} {inst} published findings confirming that {brand}'s {product} {action} {subst}. Testing was conducted across three independent batches.",
            "{brand} released its official quality report showing consistent {subst} {action} across all product lines.",
            "Independent analysis at the {city} lab verified the {subst} claim for {brand}'s {product}, with results reproduced twice.",
            "A statement from the {city} {inst} confirmed the finding, citing data from {metric} separate test sites.",
        ],
        "weak": [
            "An anonymous post on a community forum allegedly claimed that {brand}'s {product} {action} {subst}, though no official confirmation was given.",
            "Sources allegedly familiar with the matter suggest {brand}'s {product} may {action} {subst}, but the claim remains unverified.",
            "Unverified online reports from {city} indicate {brand}'s {product} possibly {action} {subst}, according to unnamed sources.",
            "A rumour circulating on social media reportedly stated that {brand}'s {product} {action} {subst} — no source was cited.",
        ],
        "hedged": [
            "A {city} report noted possible {subst} {action} in {brand}'s {product}, though the methodology was questioned by reviewers.",
            "The {inst} released preliminary data suggesting {brand}'s {product} may {action} {subst}, pending further verification.",
        ],
    },
    "perception": {
        "strong": [
            "Direct visual inspection confirms {brand}'s {product} clearly shows the described feature during normal operation.",
            "Physical examination of the {brand} {product} unit verified the observation with no ambiguity.",
            "On-site inspection at the {city} facility confirmed the described visual characteristic was present.",
            "A systematic search of the {city} facility found no instance of the described feature on any {brand} {product} unit.",
            "Comprehensive inspection confirmed the complete absence of the described characteristic from all {brand} {product} units examined.",
        ],
        "weak": [
            "A partial view reportedly suggests the {brand} {product} may display the described feature, though the angle was obstructed.",
            "An observer allegedly noted the characteristic on the {brand} {product}, but lighting conditions were poor.",
            "Photographic evidence from an unverified source purportedly shows the described feature on a {brand} {product}.",
        ],
    },
    "inference": {
        "strong": [
            "Analysis of {brand} performance logs combined with {city} sensor data indicates a clear pattern consistent with the claim.",
            "Cross-referencing {metric} data points from the {inst} study yields a conclusion strongly consistent with the claim.",
        ],
        "weak": [
            "A preliminary multi-step analysis of {brand} data from {city} suggests the claim is plausible, but intermediate steps are unverified.",
            "Initial modelling by the {inst} tentatively supports the claim, though the inference chain relies on several assumptions.",
        ],
        "hedged": [
            "Researchers at the {city} {inst} noted that available data is consistent with but does not conclusively establish the claim.",
        ],
    },
    "comparison_analogy": {
        "strong": [
            "Comparative measurements show {brand}'s {product} registers {metric} {unit} against the baseline — consistent with the stated figure.",
            "Statistical analysis of {metric} samples from {city} confirms the ratio described in the claim.",
        ],
        "weak": [
            "Rough estimates from a {city} survey suggest the comparison is approximately correct, though the sample was small ({metric} units).",
            "Informal benchmarking from an unverified report suggests a figure in the range described, but formal comparison is lacking.",
        ],
    },
    "non_apprehension": {
        "strong": [
            "A systematic {metric}-point inspection of the {city} {inst} found no instance matching the described characteristic.",
            "Comprehensive testing across all {brand} {product} batches confirmed the complete absence of {subst}.",
            "No evidence of {subst} was found in any of the {metric} samples tested across {city} facilities.",
            "A thorough search confirmed {subst} is absent from the {city} {inst} inventory.",
        ],
        "weak": [
            "A partial survey of {brand} products in {city} found no obvious instances, though not all units were examined.",
        ],
    },
}


def _ctx() -> dict:
    return {
        "brand": random.choice(_BRANDS),
        "city": random.choice(_CITIES),
        "inst": random.choice(_INSTS),
        "product": random.choice(_PRODUCTS),
        "action": random.choice(_ACTIONS),
        "subst": random.choice(_SUBSTANCES),
        "metric": random.choice(_METRICS),
        "unit": random.choice(_UNITS),
    }


def _render(tmpl: str, c: dict) -> str:
    return tmpl.format(**c)


def _pick_pool(evidence_type: str, reliability: str) -> list[str]:
    type_pools = _TEXT_POOLS.get(evidence_type) or _TEXT_POOLS["testimony"]
    return (
        type_pools.get(reliability)
        or type_pools.get("strong")
        or list(type_pools.values())[0]
    )


def _make_evidence_text(spec: EvidenceSpec, c: dict) -> str:
    pool = _pick_pool(
        spec.evidence_types[0] if spec.evidence_types else "testimony", spec.reliability
    )
    base = _render(random.choice(pool), c)
    if spec.reliability == "weak" and not base.startswith(
        tuple(p.strip() for p in _WEAK_PREFIXES)
    ):
        base = random.choice(_WEAK_PREFIXES) + base[0].lower() + base[1:]
    return base


def _make_claim(primary_type: str, c: dict) -> str:
    pool = _CLAIM_POOLS.get(primary_type) or _CLAIM_POOLS["testimony"]
    return _render(random.choice(pool), c)


class LocalTextClient(SyntheticTextClient):
    """Offline client — no API key required.

    Uses vocabulary pools with random entity substitution to produce varied
    fictional text.  All epistemic parameters are still controlled by the
    caller via EvidenceSpec; this only provides the linguistic layer.

    Usage::

        gen = FictionalClaimGenerator(_client=LocalTextClient(), registry=reg)
    """

    def generate(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        primary_type = (
            specs[0].evidence_types[0]
            if specs and specs[0].evidence_types
            else "testimony"
        )
        c = _ctx()
        claim = _make_claim(primary_type, c)
        evidence_texts = [_make_evidence_text(spec, _ctx()) for spec in specs]
        return {"claim": claim, "evidence_texts": evidence_texts}
