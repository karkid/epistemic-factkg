"""Orchestrates shortcut-breaking synthetic claim generation.

Design
------
This module owns the *epistemic* layer: template configs, EC formula
application, verdict derivation, and v3.0 record assembly.

Text generation is delegated to a pluggable SyntheticTextClient:
* LocalTextClient  — offline, template-based (no API key required)
* GroundedClient   — draws from seed_pool.jsonl for semantic coherence
* LLMClient        — calls Anthropic API for maximum linguistic diversity

Shortcut-breaking is **guaranteed by construction**: the template type
determines source trust and inference strength, which determines the
verdict through the EC formula — regardless of evidence stance text.

Template matrix and math
------------------------
All EC values use EC = 1-(1-ST)^(EW*IS).  EW values from config:
  testimony=0.80, perception=0.95, non_apprehension=0.75,
  comparison_analogy=0.65, inference=0.55.

Template                  Items                         Verdict         SB?
high_trust_supported      2 reuters testimony IS=0.8   supported       no
low_trust_nee             1 social_media IS=0.6        NEE             YES
high_trust_refuted        2 apnews testimony IS=0.8    refuted         no
low_trust_refuted_nee     1 unknown_web IS=0.5         NEE             YES
conflicting               reuters S + apnews R IS=0.8  conflicting     YES
strong_support_weak_refute  2 S IS=0.8 + 1R IS=0.4    supported       YES (has R)
weak_support_strong_refute  1S IS=0.6 + 2R IS=0.8     refuted         YES (has S)
weak_vs_weak_nee            1S IS=0.5 + 1R IS=0.4     NEE             YES (both)
corroborating_3             3 testimony IS=0.7-0.8     supported       no
perception_direct           1 ai2thor perception IS=1.0 supported      no
inference_nee               2 academic inference IS=0.5 NEE            YES (S→NEE)
comparison_supported        2 reuters comp IS=0.7      supported       no
non_apprehension_absent     1 ai2thor non_app IS=0.8   supported       no
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.adapters.synthetic.client.base import EvidenceSpec, SyntheticTextClient
from src.core.claims.labels import (
    EvidenceStance,
    EvidenceType,
    Verdict,
    aggregate_scores,
    combine_evidence_weights,
    compute_evidence_confidence,
    derive_verdict,
    get_source_trust,
    load_source_trust_registry,
)
from src.utils.time import utc_now_iso

MIN_SHORTCUT_FRACTION = 0.35


# ---------------------------------------------------------------------------
# Template configurations
# ---------------------------------------------------------------------------

@dataclass
class _TemplateConfig:
    name: str
    description: str
    evidence_specs: list[EvidenceSpec]
    is_shortcut_breaking: bool = False


_TEMPLATES: dict[str, _TemplateConfig] = {
    # ── Basic single / dual evidence ──────────────────────────────────────────
    "high_trust_supported": _TemplateConfig(
        name="high_trust_supported",
        description="Two independent, clear, direct evidence items supporting the claim.",
        evidence_specs=[
            EvidenceSpec("supports", "reuters_web_text",  ["testimony"], 0.8, "strong"),
            EvidenceSpec("supports", "apnews_web_text",   ["testimony"], 0.8, "strong"),
        ],
    ),
    "low_trust_nee": _TemplateConfig(
        name="low_trust_nee",
        description="One vague, unverified item appearing to support the claim.",
        evidence_specs=[
            EvidenceSpec("supports", "social_media_web_text", ["testimony"], 0.6, "weak"),
        ],
        is_shortcut_breaking=True,
    ),
    "high_trust_refuted": _TemplateConfig(
        name="high_trust_refuted",
        description="Two independent, clear evidence items refuting the claim.",
        evidence_specs=[
            EvidenceSpec("refutes", "reuters_web_text",  ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes", "apnews_web_text",   ["testimony"], 0.8, "strong"),
        ],
    ),
    "low_trust_refuted_nee": _TemplateConfig(
        name="low_trust_refuted_nee",
        description="One vague, uncertain item appearing to refute the claim.",
        evidence_specs=[
            EvidenceSpec("refutes", "unknown_web", ["testimony"], 0.5, "weak"),
        ],
        is_shortcut_breaking=True,
    ),
    "conflicting": _TemplateConfig(
        name="conflicting",
        description="Two reputable sources that directly contradict each other.",
        evidence_specs=[
            EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes",  "apnews_web_text",  ["testimony"], 0.8, "strong"),
        ],
        is_shortcut_breaking=True,
    ),

    # ── Compound asymmetric trust (critical for shortcut-breaking) ────────────
    "strong_support_weak_refute": _TemplateConfig(
        name="strong_support_weak_refute",
        description=(
            "Two strong supporting items and one weak refuting item. "
            "Claim is supported despite the presence of a refuting stance."
        ),
        evidence_specs=[
            EvidenceSpec("supports", "reuters_web_text",       ["testimony"], 0.8, "strong"),
            EvidenceSpec("supports", "apnews_web_text",        ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes",  "social_media_web_text",  ["testimony"], 0.4, "weak"),
        ],
        is_shortcut_breaking=True,
    ),
    "weak_support_strong_refute": _TemplateConfig(
        name="weak_support_strong_refute",
        description=(
            "One weak supporting item and two strong refuting items. "
            "Claim is refuted despite the presence of a supporting stance."
        ),
        evidence_specs=[
            EvidenceSpec("supports", "social_media_web_text", ["testimony"], 0.6, "weak"),
            EvidenceSpec("refutes",  "reuters_web_text",      ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes",  "apnews_web_text",       ["testimony"], 0.8, "strong"),
        ],
        is_shortcut_breaking=True,
    ),
    "weak_vs_weak_nee": _TemplateConfig(
        name="weak_vs_weak_nee",
        description=(
            "One weak supporting and one weak refuting item — both insufficient. "
            "Result is not_enough_evidence despite both stances being present."
        ),
        evidence_specs=[
            EvidenceSpec("supports", "unknown_web",           ["testimony"], 0.5, "weak"),
            EvidenceSpec("refutes",  "social_media_web_text", ["testimony"], 0.4, "weak"),
        ],
        is_shortcut_breaking=True,
    ),
    "corroborating_3": _TemplateConfig(
        name="corroborating_3",
        description="Three independent sources all supporting the claim.",
        evidence_specs=[
            EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong"),
            EvidenceSpec("supports", "apnews_web_text",  ["testimony"], 0.8, "strong"),
            EvidenceSpec("supports", "bbc_web_text",     ["testimony"], 0.7, "strong"),
        ],
    ),

    # ── Evidence type diversity ───────────────────────────────────────────────
    "perception_direct": _TemplateConfig(
        name="perception_direct",
        description="One direct observational item confirming the claim (simulator ground truth).",
        evidence_specs=[
            EvidenceSpec("supports", "ai2thor_simulation", ["perception"], 1.0, "strong"),
        ],
    ),
    "inference_nee": _TemplateConfig(
        name="inference_nee",
        description=(
            "Two multi-step inference items from academic sources — each moderate quality. "
            "Combined support is insufficient to reach the confirmed threshold."
        ),
        evidence_specs=[
            EvidenceSpec("supports", "academic_pdf", ["inference"], 0.5, "weak"),
            EvidenceSpec("supports", "academic_pdf", ["inference"], 0.5, "weak"),
        ],
        is_shortcut_breaking=True,  # supports stance but verdict = NEE
    ),
    "comparison_supported": _TemplateConfig(
        name="comparison_supported",
        description="Two statistical / numerical comparison items supporting the claim.",
        evidence_specs=[
            EvidenceSpec("supports", "reuters_web_text", ["comparison_analogy"], 0.7, "strong"),
            EvidenceSpec("supports", "apnews_web_text",  ["comparison_analogy"], 0.7, "strong"),
        ],
    ),
    "non_apprehension_absent": _TemplateConfig(
        name="non_apprehension_absent",
        description="One sensor-confirmed absence item supporting a negative claim.",
        evidence_specs=[
            EvidenceSpec("absent", "ai2thor_simulation", ["non_apprehension"], 0.8, "absent"),
        ],
    ),
    "non_apprehension_refuted": _TemplateConfig(
        name="non_apprehension_refuted",
        description="Confirmed absence of something expected — refuting a positive claim.",
        evidence_specs=[
            EvidenceSpec("refutes", "ai2thor_simulation", ["non_apprehension"], 0.8, "strong"),
        ],
    ),
    "non_apprehension_weak_nee": _TemplateConfig(
        name="non_apprehension_weak_nee",
        description=(
            "Weak absence evidence from an unverified source — insufficient to confirm "
            "the absence, resulting in not_enough_evidence despite absent stance."
        ),
        evidence_specs=[
            EvidenceSpec("absent", "general_web_text", ["non_apprehension"], 0.6, "weak"),
        ],
        is_shortcut_breaking=True,  # absent stance but verdict = NEE
    ),
}

_DEFAULT_DISTRIBUTION: dict[str, float] = {
    "high_trust_supported":          0.06,
    "low_trust_nee":                 0.07,
    "high_trust_refuted":            0.06,
    "low_trust_refuted_nee":         0.07,
    "conflicting":                   0.07,
    "strong_support_weak_refute":    0.09,
    "weak_support_strong_refute":    0.09,
    "weak_vs_weak_nee":              0.09,
    "corroborating_3":               0.06,
    "perception_direct":             0.07,
    "inference_nee":                 0.07,
    "comparison_supported":          0.06,
    "non_apprehension_absent":       0.03,
    "non_apprehension_refuted":      0.04,
    "non_apprehension_weak_nee":     0.07,  # SB: absent stance but NEE
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FictionalClaimGenerator:
    """Generates shortcut-breaking synthetic v3.0 claims.

    Args:
        registry:  Source trust registry dict.
        _client:   SyntheticTextClient implementation.
                   Defaults to LocalTextClient (offline, no API key).
        model:     Anthropic model ID (used only if constructing LLMClient internally).
        api_key:   Anthropic API key (used only if constructing LLMClient internally).
    """

    def __init__(
        self,
        registry: dict | None = None,
        _client: SyntheticTextClient | None = None,
        # Legacy / convenience params for direct LLM usage:
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 500,
    ):
        self._registry = registry or {}

        if _client is not None:
            self._client = _client
        elif api_key:
            from src.adapters.synthetic.llm.llm_client import LLMClient
            self._client = LLMClient(model=model, api_key=api_key, max_tokens=max_tokens)
        else:
            from src.adapters.synthetic.client.local_client import LocalTextClient
            self._client = LocalTextClient()

    def generate_batch(
        self,
        n_records: int = 100,
        distribution: dict[str, float] | None = None,
    ) -> list[dict]:
        dist = distribution or _DEFAULT_DISTRIBUTION
        plan = _make_plan(n_records, dist)
        records: list[dict] = []
        for template_key, count in plan.items():
            template = _TEMPLATES[template_key]
            for _ in range(count):
                rec = self._generate_one(template)
                if rec is not None:
                    records.append(rec)
        return records

    def _generate_one(self, template: _TemplateConfig) -> dict | None:
        result = self._client.generate(template.evidence_specs, template.name)
        if result is None:
            return None
        return _build_record(result, template, self._registry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_plan(n_records: int, distribution: dict[str, float]) -> dict[str, int]:
    """Allocate record counts per template type, guaranteed to sum to n_records.

    Uses floor allocation then distributes the remainder to the keys with the
    largest fractional parts (largest-remainder method).
    """
    counts = {k: int(n_records * f) for k, f in distribution.items()}
    remainder = n_records - sum(counts.values())
    # Sort by descending fractional part to distribute remainder fairly
    by_frac = sorted(
        distribution.keys(),
        key=lambda k: (n_records * distribution[k]) % 1,
        reverse=True,
    )
    for key in by_frac[:remainder]:
        counts[key] += 1
    return counts


def _build_record(
    parsed: dict[str, Any],
    template: _TemplateConfig,
    registry: dict,
) -> dict:
    """Assemble a complete v3.0 record from generated text + template config."""
    rec_id = f"synthetic-{uuid.uuid4().hex[:12]}"
    claim = parsed["claim"]
    ev_texts: list[str] = parsed["evidence_texts"]
    ev_triples: list[list] = parsed.get("evidence_triples") or []

    evidence_items: list[dict] = []
    for i, spec in enumerate(template.evidence_specs):
        text = ev_texts[i] if i < len(ev_texts) else ev_texts[-1]
        triples = ev_triples[i] if i < len(ev_triples) else []
        ev_id = f"{rec_id}-e{i}"

        st = get_source_trust(spec.source_id, registry)
        ew = combine_evidence_weights(spec.evidence_types)
        ec = compute_evidence_confidence(st, ew, spec.inference_strength)

        evidence_items.append({
            "evidence_id": ev_id,
            "text": text,
            "triples": triples,
            "triple_source": "ai2thor_simulation" if triples else None,
            "modality": "web_text",
            "stance": spec.stance,
            "evidence_types": list(spec.evidence_types),
            "source_id": spec.source_id,
            "inference_strength": spec.inference_strength,
            "source_url": None,
            "_ec": ec,
        })

    support_score, refute_score = aggregate_scores(evidence_items, registry)
    verdict_label = derive_verdict(support_score, refute_score)

    evidence_out = [{k: v for k, v in e.items() if not k.startswith("_")} for e in evidence_items]
    evidence_types_all = sorted({t for e in evidence_out for t in e.get("evidence_types", [])})

    return {
        "schema_version": "3.0",
        "id": rec_id,
        "claim": claim,
        "verdict": {
            "label": verdict_label.value if isinstance(verdict_label, Verdict) else verdict_label,
            "justification": None,
            "derivation_method": "aggregated_from_evidence",
        },
        "epistemic": {
            "evidence_types_all": evidence_types_all,
            "assignment_method": "synthetic_template",
        },
        "claim_triples": None,
        "reasoning": None,
        "evidence": evidence_out,
        "provenance": {
            "dataset": "synthetic",
            "split": None,
            "context_id": None,
        },
        "meta": {
            "schema_version": "3.0",
            "created_utc": utc_now_iso(),
            "template_type": template.name,
            "is_shortcut_breaking": template.is_shortcut_breaking,
        },
    }
