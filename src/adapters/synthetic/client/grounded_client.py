"""Grounded synthetic text client — seed pool + AI2THOR triplet integration.

Two source pools:
1. Text seed pool (data/registry/seed_pool.jsonl) — hand-curated fictional
   (claim, supporting_evidence, refuting_evidence) pairs for all evidence types.
   Used for testimony, inference, comparison_analogy templates.

2. AI2THOR pool (data/raw/ai2thor/claims_all.jsonl) — real simulator claims
   with structured (subject, predicate, object) triples. Used for perception
   and non_apprehension templates, giving synthetic records real triplet structure.

The client picks the right pool based on the first spec's evidence_type:
- perception / non_apprehension → prefer AI2THOR pool (gets real triples)
- everything else → text seed pool (triples stay empty)

Falls back to LocalTextClient when no matching record is found in either pool.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import EvidenceSpec, SyntheticTextClient

_TRIPLET_TYPES = frozenset({"perception", "non_apprehension"})

_WEAK_PREFIXES = [
    "Reportedly, ",
    "According to an unverified source, ",
    "Sources allegedly suggest that ",
    "An anonymous report claimed that ",
    "Unconfirmed accounts indicate that ",
]

_STRONG_CONNECTORS = [
    "Additionally, ",
    "Furthermore, ",
    "Separately, ",
    "Corroborating this, ",
    "A second independent source confirmed that ",
]


def _apply_weak(text: str) -> str:
    prefix = random.choice(_WEAK_PREFIXES)
    return prefix + text[0].lower() + text[1:]


def _apply_connector(text: str, n: int) -> str:
    if n == 0:
        return text
    connector = random.choice(_STRONG_CONNECTORS)
    return connector + text[0].lower() + text[1:]


class GroundedClient(SyntheticTextClient):
    """Generates grounded claim+evidence from curated pools.

    Args:
        seed_pool_path:  Path to seed_pool.jsonl (text evidence for non-triplet types).
        ai2thor_path:    Path to AI2THOR claims_all.jsonl (real triples for
                         perception/non_apprehension templates). Pass None to disable.
        fallback:        Client used when no matching record is found.
                         Defaults to LocalTextClient.
    """

    def __init__(
        self,
        seed_pool_path: str | Path | None = None,
        ai2thor_path: str | Path | None = None,
        fallback: SyntheticTextClient | None = None,
    ):
        # Text seed pool: {evidence_type: [record, ...]}
        self._pool: dict[str, list[dict]] = defaultdict(list)
        # AI2THOR pool: {stance: [{claim, text, triples}, ...]}
        self._ai2thor: dict[str, list[dict]] = defaultdict(list)

        if seed_pool_path is None:
            seed_pool_path = (
                Path(__file__).resolve().parents[5] / "data/registry/seed_pool.jsonl"
            )
        self._load_seed_pool(Path(seed_pool_path))

        if ai2thor_path is None:
            ai2thor_path = (
                Path(__file__).resolve().parents[5]
                / "data/raw/ai2thor/claims_all.jsonl"
            )
        self._load_ai2thor(Path(ai2thor_path))

        if fallback is None:
            from .local_client import LocalTextClient

            fallback = LocalTextClient()
        self._fallback = fallback

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_seed_pool(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self._pool[rec["evidence_type"]].append(rec)

    def _load_ai2thor(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                claim = rec.get("claim", "")
                for ev in rec.get("evidence", []):
                    triples = ev.get("triples") or []
                    if not triples:
                        continue
                    stance = ev.get("stance", "supports")
                    self._ai2thor[stance].append(
                        {
                            "claim": claim,
                            "text": ev.get("text", ""),
                            "triples": triples,
                        }
                    )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        if not specs:
            return None

        primary_type = (
            specs[0].evidence_types[0] if specs[0].evidence_types else "testimony"
        )

        if primary_type in _TRIPLET_TYPES:
            return self._generate_from_ai2thor(specs, template_name)
        else:
            return self._generate_from_seed_pool(specs, template_name)

    def _generate_from_ai2thor(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        # Pick an AI2THOR record matching the first spec's stance
        first_stance = specs[0].stance
        candidates = self._ai2thor.get(first_stance) or []
        if not candidates:
            return self._fallback.generate(specs, template_name)

        seed = random.choice(candidates)
        claim = seed["claim"]

        evidence_texts: list[str] = []
        evidence_triples: list[list] = []

        for spec in specs:
            if spec.stance == "supports":
                # Supporting evidence must come from the same record as the claim
                # so the evidence text describes the same object, not a random one.
                ev_rec = seed
            else:
                # Refuting/absent evidence may come from any record of the right stance.
                pool = self._ai2thor.get(spec.stance) or candidates
                ev_rec = random.choice(pool)

            text = ev_rec["text"]
            triples = ev_rec["triples"]

            if spec.reliability == "weak":
                text = _apply_weak(text)

            evidence_texts.append(text)
            evidence_triples.append(triples)

        return {
            "claim": claim,
            "evidence_texts": evidence_texts,
            "evidence_triples": evidence_triples,
        }

    def _generate_from_seed_pool(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        primary_type = (
            specs[0].evidence_types[0] if specs[0].evidence_types else "testimony"
        )
        candidates = self._pool.get(primary_type) or []
        if not candidates:
            return self._fallback.generate(specs, template_name)

        seed = random.choice(candidates)
        claim = seed["claim"]

        support_n = 0
        refute_n = 0
        evidence_texts: list[str] = []

        for spec in specs:
            text = self._derive_text(seed, spec, support_n, refute_n)
            if spec.stance == "supports":
                support_n += 1
            elif spec.stance == "refutes":
                refute_n += 1
            evidence_texts.append(text)

        return {"claim": claim, "evidence_texts": evidence_texts}

    def _derive_text(
        self,
        seed: dict,
        spec: EvidenceSpec,
        support_n: int,
        refute_n: int,
    ) -> str:
        stance = spec.stance
        reliability = spec.reliability

        if stance == "supports":
            base = seed.get("supporting_evidence", "")
            if reliability == "weak":
                base = _apply_weak(base)
            elif support_n > 0:
                base = _apply_connector(base, support_n)
        elif stance == "refutes":
            base = seed.get("refuting_evidence", "")
            if reliability == "weak":
                base = _apply_weak(base)
            elif refute_n > 0:
                base = _apply_connector(base, refute_n)
        else:
            base = seed.get("supporting_evidence", "")
            base = _apply_weak(base) + " — the evidence is ambiguous."

        return base or seed.get("supporting_evidence", "Evidence unavailable.")
