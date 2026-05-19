"""Validates v3.0 unified claim records against schema and semantic constraints."""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jsonschema import validate, ValidationError
from datetime import datetime

from src.epistemic.enums import Verdict, EvidenceStance, EvidenceType
from src.epistemic.schema import CLAIM_SCHEMA
from src.utils.exceptions import ValidationError as CustomValidationError


@dataclass
class ClaimValidationIssue:
    category: str
    severity: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ClaimValidationResult:
    claim_id: str
    is_valid: bool
    issues: List[ClaimValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0
    processing_time_ms: float = 0.0
    schema_valid: bool = True
    semantic_valid: bool = True
    quality_valid: bool = True
    consistency_valid: bool = True

    def add_issue(
        self,
        category: str,
        severity: str,
        message: str,
        field: str = None,
        suggestion: str = None,
    ):
        self.issues.append(
            ClaimValidationIssue(category, severity, message, field, suggestion)
        )
        if severity == "error":
            self.is_valid = False
            if category == "schema":
                self.schema_valid = False
            elif category == "semantic":
                self.semantic_valid = False
            elif category == "quality":
                self.quality_valid = False
            elif category == "consistency":
                self.consistency_valid = False


@dataclass
class AdvancedClaimValidationSummary:
    results: List[ClaimValidationResult] = field(default_factory=list)
    claim_texts: Dict[str, str] = field(default_factory=dict)

    def add_result(self, result: ClaimValidationResult):
        self.results.append(result)

    def print_summary(self):
        if not self.results:
            print("No validation results to display.")
            return

        total = len(self.results)
        valid = sum(1 for r in self.results if r.is_valid)
        avg_q = sum(r.quality_score for r in self.results) / total

        print("\n" + "=" * 60)
        print("CLAIM VALIDATION SUMMARY (schema v3.0)")
        print("=" * 60)
        print(f"Total:    {total:,}")
        print(f"Valid:    {valid:,} ({valid / total * 100:.1f}%)")
        print(f"Invalid:  {total - valid:,} ({(total - valid) / total * 100:.1f}%)")
        print(f"Avg quality score: {avg_q:.3f}")

        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        for r in self.results:
            for i in r.issues:
                by_cat[i.category] = by_cat.get(i.category, 0) + 1
                by_sev[i.severity] = by_sev.get(i.severity, 0) + 1

        if by_sev:
            print("\nIssues by severity:")
            for sev in ("error", "warning", "info"):
                if sev in by_sev:
                    print(f"  {sev}: {by_sev[sev]}")

        if by_cat:
            print("\nIssues by category:")
            for cat, n in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {n}")

        quality_ranges = {
            "Excellent (>=0.9)": 0,
            "Good (0.7-0.9)": 0,
            "Fair (0.5-0.7)": 0,
            "Poor (<0.5)": 0,
        }
        for r in self.results:
            if r.quality_score >= 0.9:
                quality_ranges["Excellent (>=0.9)"] += 1
            elif r.quality_score >= 0.7:
                quality_ranges["Good (0.7-0.9)"] += 1
            elif r.quality_score >= 0.5:
                quality_ranges["Fair (0.5-0.7)"] += 1
            else:
                quality_ranges["Poor (<0.5)"] += 1

        print("\nQuality distribution:")
        for label, count in quality_ranges.items():
            print(f"  {label}: {count:,} ({count / total * 100:.1f}%)")

        print("=" * 60)


class AdvancedClaimValidator:
    """Validates unified v3.0 claim records."""

    def __init__(self):
        self._sentence_end = re.compile(r"[.!?]$")
        self._common = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "will",
            "would",
            "can",
            "could",
            "should",
            "not",
        }

    def validate_claim_advanced(
        self, claim_json: dict, claim_id: str = None
    ) -> ClaimValidationResult:
        start = time.time()
        cid = claim_id or claim_json.get("id", "unknown")
        result = ClaimValidationResult(claim_id=cid, is_valid=True)

        try:
            self._validate_schema(claim_json, result)
            self._validate_semantics(claim_json, result)
            self._validate_quality(claim_json, result)
            self._validate_consistency(claim_json, result)
            result.quality_score = self._quality_score(claim_json, result)
        except Exception as e:
            result.add_issue("system", "error", f"Validator error: {e}")

        result.processing_time_ms = (time.time() - start) * 1000
        return result

    def _validate_schema(self, claim: dict, result: ClaimValidationResult):
        try:
            validate(instance=claim, schema=CLAIM_SCHEMA)
        except ValidationError as e:
            result.add_issue(
                "schema",
                "error",
                f"Schema: {e.message}",
                field=str(e.path) if e.path else None,
            )

    def _validate_semantics(self, claim: dict, result: ClaimValidationResult):
        triples = claim.get("claim_triples") or []
        reasoning = (claim.get("reasoning") or {}).get("structural")
        text = claim.get("claim", "")

        if reasoning == "one_hop" and len(triples) != 1:
            result.add_issue(
                "semantic",
                "error",
                f"one_hop requires exactly 1 claim triple, found {len(triples)}",
                "claim_triples",
            )

        if reasoning == "conjunction" and len(triples) < 2:
            result.add_issue(
                "semantic",
                "error",
                f"conjunction requires >=2 claim triples, found {len(triples)}",
                "claim_triples",
            )

        if reasoning == "negation":
            verdict_label = (claim.get("verdict") or {}).get("label")
            has_neg = any(
                w in text.lower() for w in ("not", "false", "n't", "never", "no ")
            )
            if not has_neg and verdict_label != "refuted":
                result.add_issue(
                    "semantic",
                    "warning",
                    "negation claim has no negation word in text",
                    "claim",
                )

        for i, triple in enumerate(triples):
            if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                result.add_issue(
                    "semantic",
                    "error",
                    f"claim_triples[{i}] malformed — expected [s, p, o]",
                    f"claim_triples[{i}]",
                )

        for ev in claim.get("evidence") or []:
            ev_types = ev.get("evidence_types") or []
            ev_id = ev.get("evidence_id", "?")
            is_non_apprehension = "non_apprehension" in ev_types
            ev_stance = ev.get("stance")
            ev_modality = ev.get("modality", "")

            if (
                is_non_apprehension
                and ev_stance == "supports"
                and (ev.get("triples") or [])
            ):
                result.add_issue(
                    "semantic",
                    "warning",
                    f"evidence {ev_id}: non_apprehension supports stance but triples is non-empty",
                    "evidence[].triples",
                    suggestion="supported absence claims should have triples=[]",
                )

            # evidence.text must not be whitespace-only
            raw_text = ev.get("text", "")
            if isinstance(raw_text, str) and raw_text and not raw_text.strip():
                result.add_issue(
                    "semantic",
                    "error",
                    f"evidence {ev_id}: text is whitespace-only",
                    "evidence[].text",
                )

            # non-unanswerable evidence must carry at least one evidence_type
            if ev_modality != "unanswerable" and not ev_types:
                result.add_issue(
                    "semantic",
                    "error",
                    f"evidence {ev_id}: evidence_types is empty for non-unanswerable evidence",
                    "evidence[].evidence_types",
                )

    def _validate_quality(self, claim: dict, result: ClaimValidationResult):
        text = claim.get("claim", "").strip()
        if not text:
            result.add_issue("quality", "error", "Claim text is empty", "claim")
            return

        if len(text) < 10:
            result.add_issue(
                "quality",
                "warning",
                f"Claim text very short ({len(text)} chars)",
                "claim",
            )
        if len(text) > 250:
            result.add_issue(
                "quality",
                "warning",
                f"Claim text very long ({len(text)} chars)",
                "claim",
            )
        if not self._sentence_end.search(text):
            result.add_issue(
                "quality", "warning", "Claim does not end with punctuation", "claim"
            )
        if not text[0].isupper():
            result.add_issue(
                "quality",
                "warning",
                "Claim does not start with capital letter",
                "claim",
            )

        words = text.lower().split()
        if len(words) < 3:
            result.add_issue(
                "quality",
                "warning",
                f"Claim has very few words ({len(words)})",
                "claim",
            )

        meaningful = [w for w in words if w not in self._common and len(w) > 2]
        if len(meaningful) < 2:
            result.add_issue(
                "quality", "warning", "Claim may lack meaningful content", "claim"
            )

        if "http://" in text or "https://" in text:
            result.add_issue(
                "quality",
                "error",
                "Claim text contains raw URIs — should be human-readable",
                "claim",
            )

        if "|" in text and "%" in text:
            result.add_issue(
                "quality",
                "warning",
                "Claim may contain unconverted entity IDs",
                "claim",
            )

        for bad, suggestion in [
            ("is temperature", "specify value: hot/cold/at room temperature"),
            (" is true.", "boolean should not show 'true'"),
            (" is false.", "boolean should not show 'false'"),
        ]:
            if bad in text.lower():
                result.add_issue(
                    "quality",
                    "warning",
                    f"Awkward phrasing: '{bad}'",
                    "claim",
                    suggestion=suggestion,
                )

    def _validate_consistency(self, claim: dict, result: ClaimValidationResult):
        label = (claim.get("verdict") or {}).get("label")
        evidence = claim.get("evidence") or []
        claim_triples_raw = claim.get("claim_triples")
        evidence_types_all = (claim.get("epistemic") or {}).get(
            "evidence_types_all", []
        )
        is_absence = EvidenceType.NON_APPREHENSION.value in evidence_types_all

        if label and label not in [v.value for v in Verdict]:
            result.add_issue(
                "consistency",
                "error",
                f"Unknown verdict label: {label}",
                "verdict.label",
            )

        # evidence_types_all must equal the sorted union of per-evidence evidence_types
        computed_union = sorted({
            t
            for e in evidence
            for t in (e.get("evidence_types") or [])
        })
        if computed_union != sorted(evidence_types_all):
            result.add_issue(
                "consistency",
                "error",
                f"evidence_types_all {evidence_types_all!r} does not match union of per-evidence types {computed_union!r}",
                "epistemic.evidence_types_all",
            )

        stances = [e.get("stance") for e in evidence if e.get("stance")]
        # verdict=supported requires at least one supports stance
        if label == Verdict.SUPPORTED.value and stances and not any(
            s == EvidenceStance.SUPPORTS.value for s in stances
        ):
            result.add_issue(
                "consistency",
                "error",
                "verdict is 'supported' but no evidence item has stance='supports'",
                "evidence[].stance",
            )
        # verdict=refuted requires at least one refutes stance
        if label == Verdict.REFUTED.value and stances and not any(
            s == EvidenceStance.REFUTES.value for s in stances
        ):
            result.add_issue(
                "consistency",
                "error",
                "verdict is 'refuted' but no evidence item has stance='refutes'",
                "evidence[].stance",
            )
        if label == Verdict.CONFLICTING_EVIDENCE and stances:
            has_supports = any(s == EvidenceStance.SUPPORTS for s in stances)
            has_refutes = any(s == EvidenceStance.REFUTES for s in stances)
            if not (has_supports or has_refutes):
                result.add_issue(
                    "consistency",
                    "warning",
                    "conflicting_evidence verdict but no supports/refutes stances found in evidence",
                    "evidence[].stance",
                )

        if label == Verdict.NOT_ENOUGH_EVIDENCE and is_absence:
            result.add_issue(
                "consistency",
                "error",
                "non_apprehension cannot have verdict not_enough_evidence — absence is a definite answer",
                "verdict.label",
            )

        if is_absence:
            decisive_stances = {EvidenceStance.SUPPORTS.value, EvidenceStance.REFUTES.value}
            has_decisive = any(e.get("stance") in decisive_stances for e in evidence)
            if not has_decisive:
                result.add_issue(
                    "consistency",
                    "error",
                    "non_apprehension evidence type requires at least one evidence item with stance=supports or stance=refutes",
                    "evidence[].stance",
                )

        meta_utc = (claim.get("meta") or {}).get("created_utc")
        if meta_utc:
            try:
                datetime.fromisoformat(meta_utc.replace("Z", "+00:00"))
            except ValueError, TypeError:
                result.add_issue(
                    "consistency",
                    "warning",
                    "meta.created_utc is not a valid ISO timestamp",
                    "meta.created_utc",
                )

        provenance = claim.get("provenance") or {}
        dataset = provenance.get("dataset", "")
        structural = (claim.get("reasoning") or {}).get("structural")
        if (
            dataset == "ai2thor"
            and claim_triples_raw is None
            and not is_absence
            and structural != "absence"
        ):
            result.add_issue(
                "consistency",
                "warning",
                "AI2THOR record missing claim_triples",
                "claim_triples",
            )
        if dataset == "averitec" and claim_triples_raw is not None:
            result.add_issue(
                "consistency",
                "info",
                "AVeriTeC record unexpectedly has claim_triples",
                "claim_triples",
            )

    def _quality_score(self, claim: dict, result: ClaimValidationResult) -> float:
        score = 1.0
        for issue in result.issues:
            if issue.severity == "error":
                score -= 0.3
            elif issue.severity == "warning":
                score -= 0.1
            elif issue.severity == "info":
                score -= 0.02

        text = claim.get("claim", "")
        if 10 <= len(text) <= 150:
            score += 0.05
        if self._sentence_end.search(text):
            score += 0.02
        if len(text.split()) >= 5:
            score += 0.03

        return max(0.0, min(1.0, score))


def semantic_checks(claim: dict):
    """Legacy helper — kept for backward compatibility."""
    result = AdvancedClaimValidator().validate_claim_advanced(claim)
    errors = [
        i.message
        for i in result.issues
        if i.severity == "error" and i.category == "semantic"
    ]
    if errors:
        raise ValueError(errors[0])


def validate_claim(claim_json: dict) -> bool:
    """Legacy helper — kept for backward compatibility."""
    result = AdvancedClaimValidator().validate_claim_advanced(claim_json)
    if not result.is_valid:
        for issue in result.issues:
            if issue.severity == "error":
                if issue.category == "schema":
                    raise CustomValidationError(
                        f"Claim failed schema validation: {issue.message}"
                    )
                raise ValueError(issue.message)
    return True
