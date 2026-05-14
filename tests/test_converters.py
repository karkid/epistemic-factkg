"""
Converter tests using permanent fixtures sampled from real data.

Fixtures (committed to tests/fixtures/):
  ai2thor_sample.jsonl  — 12 records: all structural types, both verdicts, 6 scenes
  averitec_sample.json  — 10 records: all verdict classes, diverse modalities
"""

import json
from pathlib import Path

import pytest

from src.adapters.ai2thor.converter import AI2ThorConverter
from src.adapters.averitec.converter import AveritecConverter, _infer_pramana
from src.core.claims.labels import (
    combine_pramana_weights,
    EvidenceStance,
    EvidenceType,
    Pramana,
    Verdict,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ai2thor_raw() -> list[dict]:
    with open(FIXTURES / "ai2thor_sample.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def averitec_raw() -> list[dict]:
    with open(FIXTURES / "averitec_sample.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ai2thor_converted(ai2thor_raw) -> list[dict]:
    conv = AI2ThorConverter()
    return [conv.convert_one(r, r["id"]) for r in ai2thor_raw]


@pytest.fixture(scope="module")
def averitec_converted(averitec_raw) -> list[dict]:
    conv = AveritecConverter()
    return [
        conv.convert_one(r, f"averitec-train-{i:06d}")
        for i, r in enumerate(averitec_raw, 1)
    ]


# ---------------------------------------------------------------------------
# AI2THOR converter tests
# ---------------------------------------------------------------------------


class TestAI2ThorConverter:
    def test_all_records_convert(self, ai2thor_converted):
        assert len(ai2thor_converted) == 12

    def test_schema_version(self, ai2thor_converted):
        for r in ai2thor_converted:
            assert r["schema_version"] == "3.0"

    def test_required_top_level_fields(self, ai2thor_converted):
        required = {
            "id",
            "claim",
            "verdict",
            "epistemic",
            "evidence",
            "provenance",
            "meta",
        }
        for r in ai2thor_converted:
            assert required <= r.keys(), f"Missing fields in {r['id']}"

    def test_verdict_labels_are_valid(self, ai2thor_converted):
        valid = {v.value for v in Verdict}
        for r in ai2thor_converted:
            assert r["verdict"]["label"] in valid

    def test_verdict_has_derivation_method_annotated(self, ai2thor_converted):
        for r in ai2thor_converted:
            assert r["verdict"]["derivation_method"] == "annotated", (
                f"{r['id']}: expected derivation_method=annotated"
            )

    def test_evidence_types_all_are_valid(self, ai2thor_converted):
        valid = {p.value for p in EvidenceType}
        for r in ai2thor_converted:
            for et in r["epistemic"]["evidence_types_all"]:
                assert et in valid, (
                    f"{r['id']}: unexpected evidence_type {et!r}"
                )

    def test_ai2thor_evidence_types_limited_to_perception_and_non_apprehension(
        self, ai2thor_converted
    ):
        allowed = {EvidenceType.PERCEPTION.value, EvidenceType.NON_APPREHENSION.value}
        for r in ai2thor_converted:
            for et in r["epistemic"]["evidence_types_all"]:
                assert et in allowed, (
                    f"{r['id']}: unexpected evidence_type {et!r}"
                )

    def test_dataset_provenance(self, ai2thor_converted):
        for r in ai2thor_converted:
            assert r["provenance"]["dataset"] == "ai2thor"

    def test_evidence_stance_is_valid(self, ai2thor_converted):
        valid = {s.value for s in EvidenceStance} | {None}
        for r in ai2thor_converted:
            for ev in r.get("evidence") or []:
                assert ev.get("stance") in valid, (
                    f"{r['id']}: invalid stance {ev.get('stance')!r}"
                )

    def test_evidence_items_have_v3_fields(self, ai2thor_converted):
        for r in ai2thor_converted:
            for ev in r.get("evidence") or []:
                assert "evidence_types" in ev, f"{r['id']}: missing evidence_types"
                assert "source_id" in ev, f"{r['id']}: missing source_id"
                assert "inference_strength" in ev, f"{r['id']}: missing inference_strength"

    def test_evidence_source_id_is_ai2thor_simulation(self, ai2thor_converted):
        for r in ai2thor_converted:
            for ev in r.get("evidence") or []:
                assert ev["source_id"] == "ai2thor_simulation", (
                    f"{r['id']}: unexpected source_id {ev['source_id']!r}"
                )

    def test_evidence_inference_strength_is_1(self, ai2thor_converted):
        for r in ai2thor_converted:
            for ev in r.get("evidence") or []:
                assert ev["inference_strength"] == 1.0, (
                    f"{r['id']}: expected IS=1.0, got {ev['inference_strength']}"
                )

    def test_refuted_claims_have_refutes_stance(self, ai2thor_converted):
        for r in ai2thor_converted:
            if r["verdict"]["label"] == Verdict.REFUTED.value:
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert EvidenceStance.REFUTES.value in stances, (
                    f"{r['id']}: refuted but no refutes stance"
                )

    def test_supported_perception_claims_have_supports_stance(self, ai2thor_converted):
        for r in ai2thor_converted:
            ets = r["epistemic"]["evidence_types_all"]
            v = r["verdict"]["label"]
            is_absence = EvidenceType.NON_APPREHENSION.value in ets
            if (
                v == Verdict.SUPPORTED.value
                and EvidenceType.PERCEPTION.value in ets
                and not is_absence
            ):
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert EvidenceStance.SUPPORTS.value in stances, (
                    f"{r['id']}: perception+supported but no supports stance"
                )

    def test_absence_claims(self, ai2thor_converted):
        absence = [
            r
            for r in ai2thor_converted
            if EvidenceType.NON_APPREHENSION.value in r["epistemic"]["evidence_types_all"]
        ]
        assert len(absence) >= 1, "No absence claims in fixture"
        for r in absence:
            # Absence claims must have stance=absent and no evidence triples
            for ev in r.get("evidence") or []:
                assert ev.get("stance") == EvidenceStance.ABSENT.value, (
                    f"{r['id']}: non_apprehension but stance={ev.get('stance')!r}"
                )
                assert ev.get("triples") == [], (
                    f"{r['id']}: absence claim has non-empty triples"
                )
            # claim_triples must be None for absence
            assert r.get("claim_triples") is None, (
                f"{r['id']}: absence claim should have claim_triples=null"
            )

    def test_conjunction_claims_have_multiple_triples(self, ai2thor_converted):
        conj = [
            r
            for r in ai2thor_converted
            if (r.get("reasoning") or {}).get("structural") == "conjunction"
        ]
        assert len(conj) >= 1, "No conjunction claims in fixture"
        for r in conj:
            ct = r.get("claim_triples") or []
            assert len(ct) >= 2, f"{r['id']}: conjunction claim has {len(ct)} triples"

    def test_uris_are_decoded(self, ai2thor_converted):
        """Percent-encoding like %7C should be decoded to | in triples."""
        for r in ai2thor_converted:
            for triple in r.get("claim_triples") or []:
                for part in triple:
                    assert "%" not in part, f"Un-decoded URI in {r['id']}: {part}"

    def test_reasoning_strategy_set_on_non_absence(self, ai2thor_converted):
        for r in ai2thor_converted:
            ets = r["epistemic"]["evidence_types_all"]
            if EvidenceType.NON_APPREHENSION.value not in ets:
                reasoning = r.get("reasoning") or {}
                assert reasoning.get("structural") is not None, (
                    f"{r['id']}: non-absence record has no reasoning.structural"
                )


# ---------------------------------------------------------------------------
# AVeriTeC converter tests
# ---------------------------------------------------------------------------


class TestAveritecConverter:
    def test_all_records_convert(self, averitec_converted):
        assert len(averitec_converted) == 10

    def test_schema_version(self, averitec_converted):
        for r in averitec_converted:
            assert r["schema_version"] == "2.0"

    def test_required_top_level_fields(self, averitec_converted):
        required = {
            "id",
            "claim",
            "verdict",
            "epistemic",
            "evidence",
            "provenance",
            "meta",
        }
        for r in averitec_converted:
            assert required <= r.keys()

    def test_verdict_labels_are_valid(self, averitec_converted):
        valid = {v.value for v in Verdict}
        for r in averitec_converted:
            assert r["verdict"]["label"] in valid

    def test_pramana_is_enum_value(self, averitec_converted):
        valid = {p.value for p in Pramana}
        for r in averitec_converted:
            assert r["epistemic"]["pramana_primary"] in valid

    def test_no_non_apprehension_pramana(self, averitec_converted):
        for r in averitec_converted:
            assert (
                r["epistemic"]["pramana_primary"] != Pramana.NON_APPREHENSION.value
            ), f"AVeriTeC should never assign non_apprehension: {r['id']}"

    def test_claim_triples_is_null(self, averitec_converted):
        for r in averitec_converted:
            assert r.get("claim_triples") is None, (
                f"AVeriTeC should have claim_triples=null: {r['id']}"
            )

    def test_dataset_provenance(self, averitec_converted):
        for r in averitec_converted:
            assert r["provenance"]["dataset"] == "averitec"

    def test_id_fallback_when_raw_has_no_id(self, averitec_raw, averitec_converted):
        """Raw AVeriTeC records have no id field; converter must assign a stable fallback."""
        for r in averitec_raw:
            assert not r.get("id"), "Expected no id in raw AVeriTeC record"
        for r in averitec_converted:
            assert r["id"], f"Converted record missing id: {r}"

    def test_supported_claims_have_supports_stance(self, averitec_converted):
        for r in averitec_converted:
            if r["verdict"]["label"] == Verdict.SUPPORTED.value:
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert EvidenceStance.SUPPORTS.value in stances, (
                    f"{r['id']}: supported but no supports stance"
                )

    def test_refuted_claims_have_refutes_stance(self, averitec_converted):
        for r in averitec_converted:
            if r["verdict"]["label"] == Verdict.REFUTED.value:
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert EvidenceStance.REFUTES.value in stances, (
                    f"{r['id']}: refuted but no refutes stance"
                )

    def test_conflicting_and_nee_stances_are_null(self, averitec_converted):
        ambiguous = {
            Verdict.CONFLICTING_EVIDENCE.value,
            Verdict.NOT_ENOUGH_EVIDENCE.value,
        }
        for r in averitec_converted:
            if r["verdict"]["label"] in ambiguous:
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert all(s is None for s in stances), (
                    f"{r['id']}: ambiguous verdict should have null stances, got {stances}"
                )

    def test_evidence_is_non_empty_list(self, averitec_converted):
        for r in averitec_converted:
            assert isinstance(r.get("evidence"), list)
            assert len(r["evidence"]) >= 1, f"{r['id']}: no evidence items"

    def test_pramana_priority_perception_beats_testimony(self):
        """When both video (perception) and web_text (testimony) are present,
        pramana_primary must be perception."""
        primary, all_p, _ = _infer_pramana(
            {"video", "web_text"}, {"url1", "url2"}, ["extractive"], "text"
        )
        assert primary == Pramana.PERCEPTION.value
        assert Pramana.TESTIMONY.value in all_p

    def test_pramana_inference_requires_one_abstractive_two_urls(self):
        # 1 abstractive + 2 URLs — should be inference (threshold: >=1 abstractive, >=2 urls)
        p1, _, _ = _infer_pramana(
            {"web_text"}, {"url1", "url2"}, ["abstractive"], "text"
        )
        assert p1 == Pramana.INFERENCE.value

        # 2 abstractive + 2 URLs — still inference
        p2, _, _ = _infer_pramana(
            {"web_text"}, {"url1", "url2"}, ["abstractive", "abstractive"], "text"
        )
        assert p2 == Pramana.INFERENCE.value

        # 1 abstractive + only 1 URL — NOT inference (URL guard prevents single-source)
        p3, _, _ = _infer_pramana({"web_text"}, {"url1"}, ["abstractive"], "text")
        assert p3 != Pramana.INFERENCE.value

        # 0 abstractive + 2 URLs — NOT inference
        p4, _, _ = _infer_pramana(
            {"web_text"}, {"url1", "url2"}, ["extractive", "extractive"], "text"
        )
        assert p4 != Pramana.INFERENCE.value

    def test_pramana_numeric_cue_triggers_comparison_analogy(self):
        p, _, _ = _infer_pramana(
            {"web_text"}, {"url1"}, ["extractive"], "GDP is 50% higher"
        )
        assert p == Pramana.COMPARISON_ANALOGY.value


# ---------------------------------------------------------------------------
# combine_pramana_weights
# ---------------------------------------------------------------------------


class TestCombinePramanaWeights:
    def test_single_pramana_returns_its_own_weight(self):
        assert abs(combine_pramana_weights(["perception"]) - 0.95) < 1e-4

    def test_single_pramana_testimony(self):
        assert abs(combine_pramana_weights(["testimony"]) - 0.80) < 1e-4

    def test_multiple_pramanas_higher_than_primary_alone(self):
        single = combine_pramana_weights(["testimony"])
        combined = combine_pramana_weights(["perception", "testimony"])
        assert combined > single

    def test_diminishing_returns_formula(self):
        # 1 - (1-0.95)*(1-0.55) = 1 - 0.05*0.45 = 0.9775
        w = combine_pramana_weights(["perception", "inference"])
        assert abs(w - 0.9775) < 1e-4

    def test_three_pramanas_higher_than_two(self):
        two = combine_pramana_weights(["perception", "testimony"])
        three = combine_pramana_weights(["perception", "testimony", "inference"])
        assert three > two

    def test_custom_weights_override(self):
        custom = {"perception": 0.50, "testimony": 0.50}
        w = combine_pramana_weights(["perception", "testimony"], custom)
        assert abs(w - 0.75) < 1e-4  # 1 - 0.5*0.5

    def test_averitec_multi_pramana_higher_than_single(self):
        """Multi-pramana record (video + web_text) gets higher weight than text only."""
        _, _, w_single = _infer_pramana({"web_text"}, {"u1"}, ["extractive"], "text")
        _, _, w_multi = _infer_pramana(
            {"video", "web_text"}, {"u1"}, ["extractive"], "text"
        )
        assert w_multi > w_single

    def test_custom_epistemic_config_applied_by_converter(self):
        """AveritecConverter.__init__ with epistemic_config uses overridden weights."""
        conv_default = AveritecConverter()
        conv_custom = AveritecConverter(
            epistemic_config={"confidence_weights": {"testimony": 0.50}}
        )
        rec = {"claim": "test", "label": "supported", "questions": []}
        _, _, w_default = conv_default.infer_pramana(rec)
        _, _, w_custom = conv_custom.infer_pramana(rec)
        # Default testimony weight is 0.80; custom is 0.50 — custom must be lower
        assert w_custom < w_default

    def test_infer_pramana_perception_plus_testimony_exact_weight(self):
        """video + web_text → perception + testimony.
        1 - (1-0.95)*(1-0.80) = 1 - 0.05*0.20 = 0.99
        """
        primary, all_p, w = _infer_pramana(
            {"video", "web_text"}, {"u1"}, ["extractive"], "plain text"
        )
        assert primary == Pramana.PERCEPTION.value
        assert Pramana.TESTIMONY.value in all_p
        assert abs(w - 0.99) < 1e-4

    def test_infer_pramana_testimony_plus_comparison_exact_weight(self):
        """web_text + numeric cue → comparison_analogy + testimony.
        1 - (1-0.65)*(1-0.80) = 1 - 0.35*0.20 = 0.93
        """
        primary, all_p, w = _infer_pramana(
            {"web_text"}, {"u1"}, ["extractive"], "GDP is 50% higher"
        )
        assert primary == Pramana.COMPARISON_ANALOGY.value
        assert Pramana.TESTIMONY.value in all_p
        assert abs(w - 0.93) < 1e-4

    def test_infer_pramana_three_types_exact_weight(self):
        """video + web_text + numeric cue → perception + comparison_analogy + testimony.
        1 - (1-0.95)*(1-0.65)*(1-0.80) = 1 - 0.05*0.35*0.20 = 1 - 0.0035 = 0.9965
        """
        primary, all_p, w = _infer_pramana(
            {"video", "web_text"}, {"u1"}, ["extractive"], "GDP is 50% higher"
        )
        assert primary == Pramana.PERCEPTION.value
        assert len(all_p) == 3
        assert abs(w - 0.9965) < 1e-4

    def test_convert_one_multi_pramana_weight_in_output_record(self):
        """AveritecConverter.convert_one sets confidence_weight for comparison_analogy + testimony."""
        conv = AveritecConverter()
        raw = {
            "claim": "GDP grew by 50 percent last year.",
            "label": "supported",
            "questions": [
                {
                    "question": "Did GDP grow by 50%?",
                    "answers": [
                        {
                            "answer": "Yes, GDP rose 50% according to the report.",
                            "answer_type": "extractive",
                            "source_medium": "web_text",
                            "source_url": "http://example.com/report",
                        }
                    ],
                }
            ],
        }
        record = conv.convert_one(raw, "test-multi-001")
        w = record["epistemic"]["confidence_weight"]
        primary = record["epistemic"]["pramana_primary"]
        all_p = record["epistemic"]["pramana_all"]
        assert primary == Pramana.COMPARISON_ANALOGY.value
        assert Pramana.TESTIMONY.value in all_p
        assert abs(w - 0.93) < 1e-4
