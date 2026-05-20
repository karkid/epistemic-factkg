"""Compact, parametric prompt builder for LLM-based synthetic generation.

Design goals
------------
* Deterministic formatting — same spec list → same prompt structure.
* Stance + reliability explicitly stated — LLM does not invent epistemic params.
* Evidence-type guidance per item — LLM knows HOW to write each text.
* JSON-only output — no markdown, no prose outside the JSON block.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.adapters.synthetic.client.base import EvidenceSpec

_RELIABILITY_GUIDE: dict[str, str] = {
    "strong": "direct, specific, verifiable — no hedging language",
    "weak": "vague, uncertain — use 'reportedly', 'allegedly', 'sources suggest', 'possibly'",
    "hedged": "exists but is ambiguous — partial coverage or internally uncertain",
    "not_enough_evidence": "insufficient evidence — 'inconclusive', 'could not be confirmed', 'evidence is lacking'",
}

_EVIDENCE_TYPE_GUIDE: dict[str, str] = {
    "testimony": "reported by a source (news article, document, spokesperson, official record)",
    "perception": "direct observation or physical measurement",
    "inference": "multi-step reasoning drawn from other observed facts",
    "comparison_analogy": "numerical or statistical comparison against a baseline or reference group",
    "non_apprehension": "confirmed absence of something expected",
    "postulation_derivation": "hypothetical or speculative reasoning",
}

_SYSTEM_PROMPT = """\
You are generating training data for an epistemic fact-checking research system.
Generate short, fictional claims and evidence about everyday household, urban, or consumer scenarios.
Use ONLY fictional entities — no real people, companies, places, or organisations.
Use made-up brand names (e.g. 'Stellark', 'Nexovac'), fictional cities (e.g. 'Northaven', 'Crestford'), \
fictional institutions, etc.
Keep claims simple and specific. Keep each evidence text concise (1–3 sentences).
"""


def build_prompt(specs: list[EvidenceSpec], template_name: str) -> str:
    n = len(specs)
    spec_lines = []
    for i, s in enumerate(specs, 1):
        ev_type = s.evidence_types[0] if s.evidence_types else "testimony"
        reliability_desc = _RELIABILITY_GUIDE.get(s.reliability, s.reliability)
        type_desc = _EVIDENCE_TYPE_GUIDE.get(ev_type, ev_type)
        spec_lines.append(
            f"  Item {i}: evidence_type={ev_type}, stance={s.stance}, "
            f"reliability={s.reliability} ({reliability_desc})\n"
            f"           type guidance: {type_desc}"
        )

    specs_block = "\n".join(spec_lines)
    example_texts = ", ".join([f'"Evidence text {i}."' for i in range(1, n + 1)])

    has_non_apprehension = any(
        "non_apprehension" in (s.evidence_types or []) for s in specs
    )
    non_app_rule = (
        "\n- For non_apprehension evidence: name the SPECIFIC substance, item, or entity"
        " from the claim — NEVER write 'the described element', 'the described item',"
        " 'the described substance', or any other generic placeholder"
    ) if has_non_apprehension else ""

    return f"""\
Generate a fictional everyday claim and {n} evidence item(s).

Evidence specifications:
{specs_block}

Rules:
- All evidence items must be about the SAME claim
- Maintain factual consistency: if item 1 supports and item 2 refutes, \
both must reference the same specific claim assertion
- Claim must be a complete declarative sentence ending with a period
- Each evidence text must be 1–3 sentences
- Use fictional entities only (no real brands, cities, people)
- Do NOT include markdown, "Example:", or anything outside the JSON{non_app_rule}

Output ONLY valid JSON:
{{
  "claim": "A single declarative sentence.",
  "evidence_texts": [{example_texts}]
}}
"""


def parse_llm_response(text: str) -> dict[str, Any] | None:
    """Extract and validate the JSON object from an LLM response."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    claim = str(obj.get("claim") or "").strip()
    ev_texts = obj.get("evidence_texts")
    if not claim or not isinstance(ev_texts, list) or not ev_texts:
        return None

    return {"claim": claim, "evidence_texts": [str(t).strip() for t in ev_texts if t]}
