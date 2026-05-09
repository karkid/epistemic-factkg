"""
Validator tests using permanent fixtures.

Tests the AI2ThorValidator, AveritecValidator, AdvancedClaimValidator, and
validate_unified_dataset summarize_file() function against the committed fixture files.
"""

import json
from pathlib import Path

import pytest

from src.adapters.ai2thor.converter import AI2ThorConverter
from src.adapters.ai2thor.validator import AI2ThorValidator
from src.adapters.averitec.converter import AveritecConverter
from src.adapters.averitec.validator import AveritecValidator
from src.core.claims.claim_validator import AdvancedClaimValidator
from src.core.claims.labels import EvidenceStance, Pramana, Verdict

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ai2thor_converted() -> list[dict]:
    conv = AI2ThorConverter()
    with open(FIXTURES / "ai2thor_sample.jsonl") as f:
        return [
            conv.convert_one(json.loads(line), json.loads(line)["id"])
            for line in f
            if line.strip()
        ]


@pytest.fixture(scope="module")
def averitec_converted() -> list[dict]:
    conv = AveritecConverter()
    with open(FIXTURES / "averitec_sample.json") as f:
        raw = json.load(f)
    return [
        conv.convert_one(r, f"averitec-train-{i:06d}") for i, r in enumerate(raw, 1)
    ]


# ---------------------------------------------------------------------------
# AI2ThorValidator
# ---------------------------------------------------------------------------


class TestAI2ThorValidator:
    def test_all_fixture_records_pass_without_warnings(self, ai2thor_converted):
        val = AI2ThorValidator()
        for r in ai2thor_converted:
            msgs = val.check(r)
            assert not msgs, f"{r['id']}: unexpected warnings {msgs}"

    def test_absence_claim_with_non_empty_triples_is_flagged(self):
        val = AI2ThorValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.NON_APPREHENSION.value},
            "evidence": [
                {"stance": EvidenceStance.ABSENT.value, "triples": [["s", "p", "o"]]}
            ],
            "claim_triples": None,
            "reasoning": {"structural": "absence"},
        }
        msgs = val.check(bad)
        assert any("non-empty evidence triples" in m for m in msgs)

    def test_non_apprehension_without_absent_stance_is_flagged(self):
        val = AI2ThorValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.NON_APPREHENSION.value},
            "evidence": [{"stance": EvidenceStance.SUPPORTS.value, "triples": []}],
            "claim_triples": None,
            "reasoning": {"structural": "one_hop"},
        }
        msgs = val.check(bad)
        assert any("expected stance=absent" in m for m in msgs)

    def test_perception_record_missing_claim_triples_is_flagged(self):
        val = AI2ThorValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {"stance": EvidenceStance.SUPPORTS.value, "triples": [["s", "p", "o"]]}
            ],
            "claim_triples": None,
            "reasoning": {"structural": "one_hop"},
        }
        msgs = val.check(bad)
        assert any("missing claim_triples" in m for m in msgs)

    def test_absence_claim_missing_claim_triples_is_not_flagged(self):
        """Non-apprehension absence claims legitimately have claim_triples=None."""
        val = AI2ThorValidator()
        ok = {
            "epistemic": {"pramana_primary": Pramana.NON_APPREHENSION.value},
            "evidence": [{"stance": EvidenceStance.ABSENT.value, "triples": []}],
            "claim_triples": None,
            "reasoning": {"structural": "absence"},
        }
        msgs = val.check(ok)
        assert not any("missing claim_triples" in m for m in msgs)

    def test_refuted_absence_claim_missing_claim_triples_is_not_flagged(self):
        """Refuted absence claims (structural=absence, pramana=perception) have
        claim_triples=None because an absent object can't be a positive triple."""
        val = AI2ThorValidator()
        ok = {
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {"stance": EvidenceStance.REFUTES.value, "triples": [["s", "p", "o"]]}
            ],
            "claim_triples": None,
            "reasoning": {"structural": "absence"},
            "verdict": {"label": "refuted"},
        }
        msgs = val.check(ok)
        assert not any("missing claim_triples" in m for m in msgs)

    def test_invalid_pramana_is_flagged(self):
        val = AI2ThorValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.TESTIMONY.value},
            "evidence": [
                {"stance": EvidenceStance.SUPPORTS.value, "triples": [["s", "p", "o"]]}
            ],
            "claim_triples": [["s", "p", "o"]],
            "reasoning": {"structural": "one_hop"},
        }
        msgs = val.check(bad)
        assert any("unexpected pramana_primary" in m for m in msgs)

    def test_missing_reasoning_is_flagged(self):
        val = AI2ThorValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {"stance": EvidenceStance.SUPPORTS.value, "triples": [["s", "p", "o"]]}
            ],
            "claim_triples": [["s", "p", "o"]],
            "reasoning": None,
        }
        msgs = val.check(bad)
        assert any("missing reasoning" in m for m in msgs)


# ---------------------------------------------------------------------------
# AdvancedClaimValidator — negation and absence exemptions
# ---------------------------------------------------------------------------


class TestAdvancedClaimValidator:
    def test_refuted_negation_does_not_trigger_negation_warning(self):
        """AI2THOR corruption removes the negation word from refuted negation claims.
        The validator must not flag these as missing a negation word."""
        val = AdvancedClaimValidator()
        claim = {
            "id": "test-001",
            "claim": "The bowl is open.",  # negation word removed by corruption
            "verdict": {"label": "refuted"},
            "reasoning": {"structural": "negation"},
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {
                    "stance": EvidenceStance.REFUTES.value,
                    "triples": [["bowl", "isOpen", "False"]],
                }
            ],
            "claim_triples": [["bowl", "isOpen", "False"]],
            "provenance": {"dataset": "ai2thor"},
        }
        result = val.validate_claim_advanced(claim)
        negation_warnings = [
            i
            for i in result.issues
            if i.category == "semantic" and "negation word" in i.message
        ]
        assert not negation_warnings, (
            f"Unexpected negation warning on refuted claim: {negation_warnings}"
        )

    def test_supported_negation_without_negation_word_triggers_warning(self):
        """Supported negation claims should still be flagged if missing a negation word."""
        val = AdvancedClaimValidator()
        claim = {
            "id": "test-002",
            "claim": "The bowl is open.",  # missing negation word, NOT refuted
            "verdict": {"label": "supported"},
            "reasoning": {"structural": "negation"},
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {
                    "stance": EvidenceStance.SUPPORTS.value,
                    "triples": [["bowl", "isOpen", "False"]],
                }
            ],
            "claim_triples": [["bowl", "isOpen", "False"]],
            "provenance": {"dataset": "ai2thor"},
        }
        result = val.validate_claim_advanced(claim)
        negation_warnings = [
            i
            for i in result.issues
            if i.category == "semantic" and "negation word" in i.message
        ]
        assert negation_warnings, (
            "Expected negation warning on supported claim missing negation word"
        )

    def test_refuted_absence_claim_triples_null_is_not_flagged(self):
        """Refuted absence claims have structural=absence and claim_triples=None.
        The validator must not flag these as missing claim_triples."""
        val = AdvancedClaimValidator()
        claim = {
            "id": "test-003",
            "claim": "There is no bowl in the room.",
            "verdict": {"label": "refuted"},
            "reasoning": {"structural": "absence"},
            "epistemic": {"pramana_primary": Pramana.PERCEPTION.value},
            "evidence": [
                {
                    "stance": EvidenceStance.REFUTES.value,
                    "triples": [["bowl", "in", "room"]],
                }
            ],
            "claim_triples": None,
            "provenance": {"dataset": "ai2thor"},
        }
        result = val.validate_claim_advanced(claim)
        triples_warnings = [
            i
            for i in result.issues
            if "claim_triples" in (i.field or "") and i.severity in ("warning", "error")
        ]
        assert not triples_warnings, (
            f"Unexpected claim_triples warning on refuted absence: {triples_warnings}"
        )


# ---------------------------------------------------------------------------
# AveritecValidator
# ---------------------------------------------------------------------------


class TestAveritecValidator:
    def test_all_fixture_records_pass_without_warnings(self, averitec_converted):
        val = AveritecValidator()
        for r in averitec_converted:
            msgs = val.check(r)
            assert not msgs, f"{r['id']}: unexpected warnings {msgs}"

    def test_non_apprehension_pramana_is_flagged(self):
        val = AveritecValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.NON_APPREHENSION.value},
            "verdict": {"label": Verdict.SUPPORTED.value},
            "evidence": [{"stance": EvidenceStance.ABSENT.value, "text": "some text"}],
            "claim_triples": None,
        }
        msgs = val.check(bad)
        assert any("non_apprehension" in m for m in msgs)

    def test_unexpected_claim_triples_is_flagged(self):
        val = AveritecValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.TESTIMONY.value},
            "verdict": {"label": Verdict.SUPPORTED.value},
            "evidence": [{"stance": EvidenceStance.SUPPORTS.value, "text": "text"}],
            "claim_triples": [["s", "p", "o"]],
        }
        msgs = val.check(bad)
        assert any("claim_triples" in m for m in msgs)

    def test_conflicting_evidence_all_supports_is_flagged(self):
        val = AveritecValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.TESTIMONY.value},
            "verdict": {"label": Verdict.CONFLICTING_EVIDENCE.value},
            "evidence": [
                {"stance": EvidenceStance.SUPPORTS.value, "text": "text"},
                {"stance": EvidenceStance.SUPPORTS.value, "text": "text2"},
            ],
            "claim_triples": None,
        }
        msgs = val.check(bad)
        assert any("all evidence stances are 'supports'" in m for m in msgs)

    def test_all_text_null_is_flagged(self):
        val = AveritecValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.TESTIMONY.value},
            "verdict": {"label": Verdict.SUPPORTED.value},
            "evidence": [{"stance": EvidenceStance.SUPPORTS.value, "text": None}],
            "claim_triples": None,
        }
        msgs = val.check(bad)
        assert any("text is null" in m for m in msgs)

    def test_no_evidence_is_flagged(self):
        val = AveritecValidator()
        bad = {
            "epistemic": {"pramana_primary": Pramana.TESTIMONY.value},
            "verdict": {"label": Verdict.SUPPORTED.value},
            "evidence": [],
            "claim_triples": None,
        }
        msgs = val.check(bad)
        assert any("no evidence" in m for m in msgs)


# ---------------------------------------------------------------------------
# Validate_unified_dataset: summarize_file on fixture JSONL
# ---------------------------------------------------------------------------


class TestSummarizeFile:
    @pytest.fixture(scope="class")
    def ai2thor_jsonl(self, tmp_path_factory) -> Path:
        """Write converted AI2THOR fixtures to a tmp JSONL for summarize_file."""
        conv = AI2ThorConverter()
        out = tmp_path_factory.mktemp("val") / "ai2thor.jsonl"
        with open(FIXTURES / "ai2thor_sample.jsonl") as f_in, open(out, "w") as f_out:
            for line in f_in:
                r = json.loads(line)
                f_out.write(json.dumps(conv.convert_one(r, r["id"])) + "\n")
        return out

    @pytest.fixture(scope="class")
    def averitec_jsonl(self, tmp_path_factory) -> Path:
        conv = AveritecConverter()
        out = tmp_path_factory.mktemp("val") / "averitec.jsonl"
        with open(FIXTURES / "averitec_sample.json") as f_in:
            raw = json.load(f_in)
        with open(out, "w") as f_out:
            for i, r in enumerate(raw, 1):
                f_out.write(
                    json.dumps(conv.convert_one(r, f"averitec-train-{i:06d}")) + "\n"
                )
        return out

    def test_ai2thor_zero_schema_errors(self, ai2thor_jsonl):
        from jsonschema import Draft7Validator
        from src.core.claims.claim_schema import CLAIM_SCHEMA
        from src.core.claims.claim_validator import AdvancedClaimValidator
        from src.cli.validate_unified_dataset import summarize_file

        s = summarize_file(
            str(ai2thor_jsonl),
            Draft7Validator(CLAIM_SCHEMA),
            AdvancedClaimValidator(),
        )
        assert s["counts"]["schema_invalid"] == 0
        # Refuted negation claims may have no negation word (corruption removes it) — warning only
        schema_errors = [
            k
            for k in s["logic_warnings_top"]
            if "[semantic:error]" in k or "[consistency:error]" in k
        ]
        assert not schema_errors, f"Unexpected errors: {schema_errors}"

    def test_averitec_zero_schema_errors(self, averitec_jsonl):
        from jsonschema import Draft7Validator
        from src.core.claims.claim_schema import CLAIM_SCHEMA
        from src.core.claims.claim_validator import AdvancedClaimValidator
        from src.cli.validate_unified_dataset import summarize_file

        s = summarize_file(
            str(averitec_jsonl),
            Draft7Validator(CLAIM_SCHEMA),
            AdvancedClaimValidator(),
        )
        assert s["counts"]["schema_invalid"] == 0
        # AVeriTeC claims often lack punctuation (web-sourced text) — quality warnings are expected
        error_keys = [k for k in s["logic_warnings_top"] if ":error]" in k]
        assert not error_keys, f"Unexpected errors: {error_keys}"

    def test_ai2thor_absence_claims_counted_correctly(self, ai2thor_jsonl):
        from jsonschema import Draft7Validator
        from src.core.claims.claim_schema import CLAIM_SCHEMA
        from src.core.claims.claim_validator import AdvancedClaimValidator
        from src.cli.validate_unified_dataset import summarize_file

        s = summarize_file(
            str(ai2thor_jsonl),
            Draft7Validator(CLAIM_SCHEMA),
            AdvancedClaimValidator(),
        )
        absence = s["gnn_readiness"]["absence_claims"]
        # Fixture has 3 absence records
        assert absence == 3

    def test_coverage_metrics_populated(self, ai2thor_jsonl):
        from jsonschema import Draft7Validator
        from src.core.claims.claim_schema import CLAIM_SCHEMA
        from src.core.claims.claim_validator import AdvancedClaimValidator
        from src.cli.validate_unified_dataset import summarize_file

        s = summarize_file(
            str(ai2thor_jsonl),
            Draft7Validator(CLAIM_SCHEMA),
            AdvancedClaimValidator(),
        )
        gnn = s["gnn_readiness"]
        assert gnn["avg_evidence_per_record"] > 0
        assert gnn["total_records"] == 12

    def test_markdown_report_is_written(self, ai2thor_jsonl, tmp_path):
        from jsonschema import Draft7Validator
        from src.core.claims.claim_schema import CLAIM_SCHEMA
        from src.core.claims.claim_validator import AdvancedClaimValidator
        from src.cli.validate_unified_dataset import (
            summarize_file,
            write_validation_report_md,
        )

        s = summarize_file(
            str(ai2thor_jsonl),
            Draft7Validator(CLAIM_SCHEMA),
            AdvancedClaimValidator(),
        )
        md_path = tmp_path / "report.md"
        write_validation_report_md([s], md_path, "2026-01-01T00:00:00Z")
        content = md_path.read_text()
        assert "# Validation Report" in content
        assert "GNN Readiness" in content
        assert "Verdict Distribution" in content
        assert "Semantic Rule Violations" in content
