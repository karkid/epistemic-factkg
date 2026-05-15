"""Synthetic data validator — checks shortcut-breaking and structural requirements."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.adapters.synthetic.fictional_generator import MIN_SHORTCUT_FRACTION
from src.core.claims.labels import (
    aggregate_scores,
    derive_verdict,
    load_source_trust_registry,
)


@dataclass
class SyntheticValidationReport:
    total: int = 0
    shortcut_breaking: int = 0
    verdict_distribution: dict[str, int] = field(default_factory=dict)
    missing_v3_fields: int = 0
    ec_mismatch: int = 0   # records where stored verdict ≠ recomputed verdict
    errors: list[str] = field(default_factory=list)

    @property
    def shortcut_fraction(self) -> float:
        return self.shortcut_breaking / self.total if self.total else 0.0

    @property
    def passes(self) -> bool:
        return (
            self.shortcut_fraction >= MIN_SHORTCUT_FRACTION
            and self.missing_v3_fields == 0
            and self.ec_mismatch == 0
            and not self.errors
        )

    def summary(self) -> str:
        lines = [
            f"Total:              {self.total}",
            f"Shortcut-breaking:  {self.shortcut_breaking} ({self.shortcut_fraction:.1%})"
            f"  [threshold: {MIN_SHORTCUT_FRACTION:.0%}]",
            f"Verdict distribution: {self.verdict_distribution}",
            f"Missing v3.0 fields: {self.missing_v3_fields}",
            f"EC mismatch:         {self.ec_mismatch}",
            f"Status:              {'PASS' if self.passes else 'FAIL'}",
        ]
        if self.errors:
            lines.append("Errors:")
            for e in self.errors[:5]:
                lines.append(f"  {e}")
        return "\n".join(lines)


_V3_EVIDENCE_REQUIRED = {"evidence_id", "text", "triples", "modality", "stance",
                          "evidence_types", "source_id", "inference_strength"}


class SyntheticDataValidator:
    """Validates a batch of synthetic v3.0 records.

    Checks:
    - ≥ MIN_SHORTCUT_FRACTION are shortcut-breaking (meta.is_shortcut_breaking)
    - All evidence items have required v3.0 fields
    - Stored verdict matches verdict derived from re-aggregating evidence EC values
      (spot-checks epistemic formula consistency)
    """

    def __init__(self, registry: dict | None = None):
        self._registry = registry or {}

    def validate_batch(self, records: list[dict]) -> SyntheticValidationReport:
        report = SyntheticValidationReport(total=len(records))
        if not records:
            report.errors.append("Empty batch — nothing to validate.")
            return report

        for rec in records:
            meta = rec.get("meta") or {}
            if meta.get("is_shortcut_breaking"):
                report.shortcut_breaking += 1

            label = (rec.get("verdict") or {}).get("label", "unknown")
            report.verdict_distribution[label] = (
                report.verdict_distribution.get(label, 0) + 1
            )

            # Check v3.0 evidence fields
            for ev in rec.get("evidence") or []:
                missing = _V3_EVIDENCE_REQUIRED - ev.keys()
                if missing:
                    report.missing_v3_fields += 1
                    report.errors.append(
                        f"{rec.get('id')}: evidence missing fields {missing}"
                    )
                    break

            # Re-derive verdict from evidence and compare
            try:
                computed_label = _recompute_verdict(rec, self._registry)
                if computed_label and computed_label != label:
                    report.ec_mismatch += 1
                    report.errors.append(
                        f"{rec.get('id')}: stored verdict={label!r}, "
                        f"recomputed={computed_label!r}"
                    )
            except Exception as exc:
                report.errors.append(f"{rec.get('id')}: recompute error: {exc}")

        if report.shortcut_fraction < MIN_SHORTCUT_FRACTION:
            report.errors.append(
                f"Shortcut fraction {report.shortcut_fraction:.1%} < "
                f"required {MIN_SHORTCUT_FRACTION:.0%}"
            )

        return report


def _recompute_verdict(rec: dict, registry: dict) -> str | None:
    evidence = rec.get("evidence") or []
    if not evidence:
        return None
    support_score, refute_score = aggregate_scores(evidence, registry)
    return derive_verdict(support_score, refute_score)


def validate_file(
    jsonl_path: str,
    registry_path: str | None = None,
) -> SyntheticValidationReport:
    """Validate a synthetic JSONL file. Convenience wrapper for CLI use."""
    import json
    from pathlib import Path

    registry: dict = {}
    if registry_path and Path(registry_path).exists():
        registry = load_source_trust_registry(registry_path)

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return SyntheticDataValidator(registry=registry).validate_batch(records)
