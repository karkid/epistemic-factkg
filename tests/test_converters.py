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
from src.adapters.averitec.converter import (
    AveritecConverter,
    _infer_evidence_types_basic,
)
from src.epistemic.enums import EvidenceStance, EvidenceType, Verdict
from src.epistemic.formula import combine_evidence_weights as combine_pramana_weights

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ai2thor_raw() -> list[dict]:
    with open(FIXTURES / "ai2thor_sample.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def averitec_raw() -> list[dict]:
    with open(FIXTURES / "averitec_sample.json", encoding="utf-8") as f:
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
                assert et in valid, f"{r['id']}: unexpected evidence_type {et!r}"

    def test_ai2thor_evidence_types_limited_to_perception_and_non_apprehension(
        self, ai2thor_converted
    ):
        allowed = {EvidenceType.PERCEPTION.value, EvidenceType.NON_APPREHENSION.value}
        for r in ai2thor_converted:
            for et in r["epistemic"]["evidence_types_all"]:
                assert et in allowed, f"{r['id']}: unexpected evidence_type {et!r}"

    def test_dataset_provenance(self, ai2thor_converted):
        for r in ai2thor_converted:
            assert r["provenance"]["dataset"] == "ai2thor"

    def test_evidence_stance_is_valid(self, ai2thor_converted):
        valid = {s.value for s in EvidenceStance}
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
                assert "inference_strength" in ev, (
                    f"{r['id']}: missing inference_strength"
                )

    def test_evidence_source_id_is_sensor_perception(self, ai2thor_converted):
        for r in ai2thor_converted:
            for ev in r.get("evidence") or []:
                assert ev["source_id"] == "sensor_perception", (
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
            if EvidenceType.NON_APPREHENSION.value
            in r["epistemic"]["evidence_types_all"]
        ]
        assert len(absence) >= 1, "No absence claims in fixture"
        for r in absence:
            # Absence claims use supports/refutes like all other claims (ADR-028)
            for ev in r.get("evidence") or []:
                assert ev.get("stance") in (
                    EvidenceStance.SUPPORTS.value,
                    EvidenceStance.REFUTES.value,
                ), f"{r['id']}: non_apprehension but stance={ev.get('stance')!r}"
                assert not ev.get("triples"), (
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
            assert r["schema_version"] == "3.0"

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

    def test_verdict_has_derivation_method_annotated(self, averitec_converted):
        for r in averitec_converted:
            assert r["verdict"]["derivation_method"] == "annotated", (
                f"{r['id']}: expected derivation_method=annotated"
            )

    def test_evidence_types_all_are_valid(self, averitec_converted):
        valid = {p.value for p in EvidenceType}
        for r in averitec_converted:
            for et in r["epistemic"]["evidence_types_all"]:
                assert et in valid, f"{r['id']}: unexpected evidence_type {et!r}"

    def test_no_non_apprehension_in_evidence_types(self, averitec_converted):
        for r in averitec_converted:
            assert (
                EvidenceType.NON_APPREHENSION.value
                not in r["epistemic"]["evidence_types_all"]
            ), f"AVeriTeC should never have non_apprehension: {r['id']}"

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

    def test_conflicting_and_nee_stances_are_not_enough_evidence(self, averitec_converted):
        ambiguous = {
            Verdict.CONFLICTING_EVIDENCE.value,
            Verdict.NOT_ENOUGH_EVIDENCE.value,
        }
        for r in averitec_converted:
            if r["verdict"]["label"] in ambiguous:
                stances = [e.get("stance") for e in r.get("evidence") or []]
                assert all(s == EvidenceStance.NOT_ENOUGH_EVIDENCE.value for s in stances), (
                    f"{r['id']}: ambiguous verdict should have not_enough_evidence stances, got {stances}"
                )

    def test_evidence_is_non_empty_list(self, averitec_converted):
        for r in averitec_converted:
            assert isinstance(r.get("evidence"), list)
            assert len(r["evidence"]) >= 1, f"{r['id']}: no evidence items"

    def test_evidence_items_have_v3_fields(self, averitec_converted):
        for r in averitec_converted:
            for ev in r.get("evidence") or []:
                assert "evidence_types" in ev, f"{r['id']}: missing evidence_types"
                assert "source_id" in ev, f"{r['id']}: missing source_id"
                assert "inference_strength" in ev, (
                    f"{r['id']}: missing inference_strength"
                )

    def test_evidence_types_basic_perception_from_perceptual_modality(self):
        """Image/video/audio modalities → evidence_type=perception only (no inference added)."""
        assert _infer_evidence_types_basic("video", "extractive", "text") == [
            EvidenceType.PERCEPTION.value
        ]
        assert _infer_evidence_types_basic("image", "extractive", "text") == [
            EvidenceType.PERCEPTION.value
        ]

    def test_evidence_types_basic_testimony_for_web_text(self):
        assert EvidenceType.TESTIMONY.value in _infer_evidence_types_basic(
            "web_text", "extractive", "plain text"
        )

    def test_evidence_types_basic_web_table_gets_comparison_analogy(self):
        """web_table modality → [comparison_analogy, testimony] per labeling guide."""
        types = _infer_evidence_types_basic("web_table", "extractive", "some data")
        assert EvidenceType.COMPARISON_ANALOGY.value in types
        assert EvidenceType.TESTIMONY.value in types
        assert EvidenceType.PERCEPTION.value not in types

    def test_evidence_types_basic_comparison_analogy_on_numeric_cue(self):
        types = _infer_evidence_types_basic("web_text", "extractive", "GDP grew by 50%")
        assert EvidenceType.COMPARISON_ANALOGY.value in types
        assert EvidenceType.TESTIMONY.value in types

    def test_evidence_types_basic_unanswerable_returns_empty(self):
        assert _infer_evidence_types_basic("web_text", "unanswerable", "") == []

    def test_numerical_comparison_strategy_adds_comparison_analogy(self):
        """fact_checking_strategies=['Numerical Comparison'] → comparison_analogy added."""
        conv = AveritecConverter()
        raw = {
            "claim": "Country X has the highest GDP.",
            "label": "supported",
            "fact_checking_strategies": ["Numerical Comparison"],
            "questions": [
                {
                    "question": "What is the GDP?",
                    "answers": [
                        {
                            "answer": "GDP is 5 trillion.",
                            "answer_type": "Extractive",
                            "source_medium": "Web text",
                            "source_url": "http://example.com",
                        }
                    ],
                }
            ],
        }
        record = conv.convert_one(raw, "test-strat-001")
        ets = record["epistemic"]["evidence_types_all"]
        assert EvidenceType.COMPARISON_ANALOGY.value in ets, (
            f"Expected comparison_analogy: {ets}"
        )

    def test_consultation_strategy_adds_inference(self):
        """fact_checking_strategies=['Consultation'] → inference added to textual items."""
        conv = AveritecConverter()
        raw = {
            "claim": "The expert confirmed the finding.",
            "label": "supported",
            "fact_checking_strategies": ["Consultation"],
            "questions": [
                {
                    "question": "What did the expert say?",
                    "answers": [
                        {
                            "answer": "Expert confirmed this is accurate.",
                            "answer_type": "Extractive",
                            "source_medium": "Web text",
                            "source_url": "http://example.com",
                        }
                    ],
                }
            ],
        }
        record = conv.convert_one(raw, "test-strat-002")
        ets = record["epistemic"]["evidence_types_all"]
        assert EvidenceType.INFERENCE.value in ets, f"Expected inference: {ets}"

    def test_strategy_enrichment_does_not_apply_to_perceptual_items(self):
        """Perceptual evidence (video) keeps only perception even with Numerical Comparison strategy."""
        conv = AveritecConverter()
        raw = {
            "claim": "The video shows the correct number.",
            "label": "supported",
            "fact_checking_strategies": ["Numerical Comparison"],
            "questions": [
                {
                    "question": "What does the video show?",
                    "answers": [
                        {
                            "answer": "The number shown is 42.",
                            "answer_type": "Extractive",
                            "source_medium": "Video",
                            "source_url": "http://example.com/video",
                        }
                    ],
                }
            ],
        }
        record = conv.convert_one(raw, "test-strat-003")
        ev = record["evidence"][0]
        assert ev["evidence_types"] == [EvidenceType.PERCEPTION.value], (
            f"Perceptual item should not be enriched: {ev['evidence_types']}"
        )

    def test_inference_added_for_abstractive_multi_source(self):
        """Abstractive items with >=2 distinct source URLs get inference type added."""
        conv = AveritecConverter()
        raw = {
            "claim": "The economy grew significantly.",
            "label": "supported",
            "questions": [
                {
                    "question": "By how much?",
                    "answers": [
                        {
                            "answer": "GDP rose 5% according to two different agencies.",
                            "answer_type": "Abstractive",
                            "source_medium": "Web text",
                            "source_url": "http://source1.com/report",
                        }
                    ],
                },
                {
                    "question": "Is this confirmed?",
                    "answers": [
                        {
                            "answer": "Yes, confirmed by the central bank.",
                            "answer_type": "Abstractive",
                            "source_medium": "Web text",
                            "source_url": "http://source2.org/data",
                        }
                    ],
                },
            ],
        }
        record = conv.convert_one(raw, "test-inf-001")
        ev_types_all = record["epistemic"]["evidence_types_all"]
        assert EvidenceType.INFERENCE.value in ev_types_all, (
            f"Expected inference in evidence_types_all: {ev_types_all}"
        )


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

    def test_perception_plus_testimony_exact_weight(self):
        # 1 - (1-0.95)*(1-0.80) = 1 - 0.05*0.20 = 0.99
        w = combine_pramana_weights(["perception", "testimony"])
        assert abs(w - 0.99) < 1e-4

    def test_comparison_analogy_plus_testimony_exact_weight(self):
        # 1 - (1-0.65)*(1-0.80) = 1 - 0.35*0.20 = 0.93
        w = combine_pramana_weights(["comparison_analogy", "testimony"])
        assert abs(w - 0.93) < 1e-4

    def test_three_types_exact_weight(self):
        # 1 - (1-0.95)*(1-0.65)*(1-0.80) = 1 - 0.05*0.35*0.20 = 0.9965
        w = combine_pramana_weights(["perception", "comparison_analogy", "testimony"])
        assert abs(w - 0.9965) < 1e-4

    def test_convert_one_evidence_types_all_for_numeric_claim(self):
        """AveritecConverter.convert_one sets evidence_types_all for comparison_analogy + testimony."""
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
        ets = record["epistemic"]["evidence_types_all"]
        assert EvidenceType.COMPARISON_ANALOGY.value in ets
        assert EvidenceType.TESTIMONY.value in ets
