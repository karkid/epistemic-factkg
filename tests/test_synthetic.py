"""Tests for synthetic generation pipeline.

All tests are offline — no real API calls.
LLMClient tests mock the Anthropic client via dependency injection.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.adapters.synthetic.client import EvidenceSpec, LocalTextClient, GroundedClient
from src.adapters.synthetic.llm import LLMClient, build_prompt, parse_llm_response
from src.adapters.synthetic.fictional_generator import (
    MIN_SHORTCUT_FRACTION,
    _TEMPLATES,
    _build_record,
    _make_plan,
    FictionalClaimGenerator,
)
from src.adapters.synthetic.validator import SyntheticDataValidator
from src.core.claims.labels import EvidenceType, Verdict, load_source_trust_registry

REGISTRY_PATH = Path(__file__).parent.parent / "data/registry/source_trust_registry.jsonl"
SEED_POOL_PATH = Path(__file__).parent.parent / "data/registry/seed_pool.jsonl"


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_source_trust_registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# _make_plan
# ---------------------------------------------------------------------------

class TestMakePlan:
    def test_total_equals_n_records(self):
        plan = _make_plan(100, {"a": 0.5, "b": 0.3, "c": 0.2})
        assert sum(plan.values()) == 100

    def test_small_n(self):
        plan = _make_plan(5, {"a": 0.6, "b": 0.4})
        assert sum(plan.values()) == 5

    def test_all_template_keys_present(self):
        from src.adapters.synthetic.fictional_generator import _DEFAULT_DISTRIBUTION
        plan = _make_plan(100, _DEFAULT_DISTRIBUTION)
        assert set(plan.keys()) == set(_DEFAULT_DISTRIBUTION.keys())


# ---------------------------------------------------------------------------
# prompt_builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_build_prompt_contains_specs(self):
        specs = [
            EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes",  "apnews_web_text",  ["testimony"], 0.8, "strong"),
        ]
        prompt = build_prompt(specs, "conflicting")
        assert "testimony" in prompt
        assert "supports" in prompt
        assert "refutes"  in prompt
        assert "strong"   in prompt

    def test_build_prompt_n_items(self):
        specs = [EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong")]
        prompt = build_prompt(specs, "single")
        assert "1 evidence item" in prompt

    def test_parse_valid_json(self):
        text = '{"claim": "The bowl is clean.", "evidence_texts": ["The bowl is spotless."]}'
        result = parse_llm_response(text)
        assert result["claim"] == "The bowl is clean."
        assert result["evidence_texts"] == ["The bowl is spotless."]

    def test_parse_json_embedded_in_prose(self):
        text = 'Sure!\n{"claim": "Test.", "evidence_texts": ["Ev."]}\n'
        assert parse_llm_response(text) is not None

    def test_parse_missing_claim_returns_none(self):
        assert parse_llm_response('{"evidence_texts": ["text"]}') is None

    def test_parse_empty_evidence_returns_none(self):
        assert parse_llm_response('{"claim": "Test.", "evidence_texts": []}') is None

    def test_parse_invalid_json_returns_none(self):
        assert parse_llm_response("not json at all") is None


# ---------------------------------------------------------------------------
# _build_record — verdict math verified by construction
# ---------------------------------------------------------------------------

class TestBuildRecord:
    def test_high_trust_supported_verdict(self, registry):
        parsed = {"claim": "The fictional device works.", "evidence_texts": ["A confirms.", "B confirms."]}
        rec = _build_record(parsed, _TEMPLATES["high_trust_supported"], registry)
        assert rec["verdict"]["label"] == Verdict.SUPPORTED.value

    def test_low_trust_nee_verdict(self, registry):
        parsed = {"claim": "The gadget runs forever.", "evidence_texts": ["Reportedly it does."]}
        rec = _build_record(parsed, _TEMPLATES["low_trust_nee"], registry)
        assert rec["verdict"]["label"] == Verdict.NOT_ENOUGH_EVIDENCE.value

    def test_high_trust_refuted_verdict(self, registry):
        parsed = {"claim": "The building is red.", "evidence_texts": ["It's blue, A says.", "It's blue, B says."]}
        rec = _build_record(parsed, _TEMPLATES["high_trust_refuted"], registry)
        assert rec["verdict"]["label"] == Verdict.REFUTED.value

    def test_low_trust_refuted_nee_verdict(self, registry):
        parsed = {"claim": "The device uses 5W.", "evidence_texts": ["Anonymous site says maybe not."]}
        rec = _build_record(parsed, _TEMPLATES["low_trust_refuted_nee"], registry)
        assert rec["verdict"]["label"] == Verdict.NOT_ENOUGH_EVIDENCE.value

    def test_conflicting_verdict(self, registry):
        parsed = {"claim": "The product is effective.", "evidence_texts": ["A says yes.", "B says no."]}
        rec = _build_record(parsed, _TEMPLATES["conflicting"], registry)
        assert rec["verdict"]["label"] == Verdict.CONFLICTING_EVIDENCE.value

    def test_strong_support_weak_refute_still_supported(self, registry):
        """Weak refuting evidence must not flip a strongly supported claim."""
        parsed = {"claim": "Claim.", "evidence_texts": ["A.", "B.", "Weak C."]}
        rec = _build_record(parsed, _TEMPLATES["strong_support_weak_refute"], registry)
        assert rec["verdict"]["label"] == Verdict.SUPPORTED.value

    def test_weak_support_strong_refute_is_refuted(self, registry):
        """Weak supporting evidence must not prevent a strong refutation."""
        parsed = {"claim": "Claim.", "evidence_texts": ["Weak A.", "B.", "C."]}
        rec = _build_record(parsed, _TEMPLATES["weak_support_strong_refute"], registry)
        assert rec["verdict"]["label"] == Verdict.REFUTED.value

    def test_weak_vs_weak_nee(self, registry):
        """Both weak stances → not_enough_evidence."""
        parsed = {"claim": "Claim.", "evidence_texts": ["Weak support.", "Weak refute."]}
        rec = _build_record(parsed, _TEMPLATES["weak_vs_weak_nee"], registry)
        assert rec["verdict"]["label"] == Verdict.NOT_ENOUGH_EVIDENCE.value

    def test_inference_nee(self, registry):
        """Two inference items at IS=0.5 should not reach the supported threshold."""
        parsed = {"claim": "Claim.", "evidence_texts": ["Inference A.", "Inference B."]}
        rec = _build_record(parsed, _TEMPLATES["inference_nee"], registry)
        assert rec["verdict"]["label"] == Verdict.NOT_ENOUGH_EVIDENCE.value

    def test_perception_direct_supported(self, registry):
        parsed = {"claim": "The cup is on the table.", "evidence_texts": ["Direct observation confirms."]}
        rec = _build_record(parsed, _TEMPLATES["perception_direct"], registry)
        assert rec["verdict"]["label"] == Verdict.SUPPORTED.value

    def test_comparison_supported(self, registry):
        parsed = {"claim": "Model A uses 40% less power.", "evidence_texts": ["Stats A.", "Stats B."]}
        rec = _build_record(parsed, _TEMPLATES["comparison_supported"], registry)
        assert rec["verdict"]["label"] == Verdict.SUPPORTED.value

    def test_no_support_score_in_verdict(self, registry):
        """support_score and refute_score are NOT stored — computed at model-build time."""
        parsed = {"claim": "Test.", "evidence_texts": ["Ev."]}
        rec = _build_record(parsed, _TEMPLATES["high_trust_supported"], registry)
        assert "support_score" not in rec["verdict"]
        assert "refute_score"  not in rec["verdict"]

    def test_record_has_required_v3_fields(self, registry):
        parsed = {"claim": "Test.", "evidence_texts": ["Evidence."]}
        rec = _build_record(parsed, _TEMPLATES["high_trust_supported"], registry)
        assert rec["schema_version"] == "3.0"
        assert rec["verdict"]["derivation_method"] == "aggregated_from_evidence"
        assert "evidence_types_all" in rec["epistemic"]
        for ev in rec["evidence"]:
            assert "evidence_types" in ev
            assert "source_id" in ev
            assert "inference_strength" in ev

    def test_shortcut_breaking_flag(self, registry):
        parsed = {"claim": "Test.", "evidence_texts": ["Ev.", "Ev2."]}
        for name, tmpl in _TEMPLATES.items():
            rec = _build_record(parsed, tmpl, registry)
            assert rec["meta"]["is_shortcut_breaking"] == tmpl.is_shortcut_breaking, name

    def test_provenance_dataset_is_synthetic(self, registry):
        parsed = {"claim": "Test.", "evidence_texts": ["Ev."]}
        rec = _build_record(parsed, _TEMPLATES["high_trust_supported"], registry)
        assert rec["provenance"]["dataset"] == "synthetic"


# ---------------------------------------------------------------------------
# LocalTextClient
# ---------------------------------------------------------------------------

class TestLocalTextClient:
    def test_returns_claim_and_evidence(self, registry):
        client = LocalTextClient()
        specs = [EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong")]
        result = client.generate(specs, "test")
        assert isinstance(result, dict)
        assert "claim" in result and "evidence_texts" in result
        assert len(result["evidence_texts"]) == 1

    def test_multi_spec_returns_matching_count(self):
        client = LocalTextClient()
        specs = [
            EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong"),
            EvidenceSpec("refutes",  "apnews_web_text",  ["testimony"], 0.8, "strong"),
        ]
        result = client.generate(specs, "conflicting")
        assert len(result["evidence_texts"]) == 2

    def test_perception_type_handled(self):
        client = LocalTextClient()
        specs = [EvidenceSpec("supports", "ai2thor_simulation", ["perception"], 1.0, "strong")]
        result = client.generate(specs, "perception_direct")
        assert result is not None

    def test_non_apprehension_absent(self):
        client = LocalTextClient()
        specs = [EvidenceSpec("absent", "ai2thor_simulation", ["non_apprehension"], 0.8, "absent")]
        result = client.generate(specs, "non_apprehension_absent")
        assert result is not None

    def test_inference_weak(self):
        client = LocalTextClient()
        specs = [
            EvidenceSpec("supports", "academic_pdf", ["inference"], 0.5, "weak"),
            EvidenceSpec("supports", "academic_pdf", ["inference"], 0.5, "weak"),
        ]
        result = client.generate(specs, "inference_nee")
        assert len(result["evidence_texts"]) == 2


# ---------------------------------------------------------------------------
# GroundedClient
# ---------------------------------------------------------------------------

class TestGroundedClient:
    def test_loads_seed_pool(self):
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH)
        assert len(client._pool) > 0

    def test_returns_grounded_claim(self):
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH)
        specs = [EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong")]
        result = client.generate(specs, "high_trust_supported")
        assert isinstance(result["claim"], str) and len(result["claim"]) > 5

    def test_weak_prefix_applied(self):
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH)
        specs = [EvidenceSpec("supports", "social_media_web_text", ["testimony"], 0.6, "weak")]
        result = client.generate(specs, "low_trust_nee")
        text = result["evidence_texts"][0].lower()
        hedging_words = ["reportedly", "allegedly", "unverified", "anonymous", "unconfirmed"]
        assert any(w in text for w in hedging_words), f"Expected hedging in: {text}"

    def test_falls_back_for_unknown_type(self):
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH)
        specs = [EvidenceSpec("supports", "ai2thor_simulation", ["perception"], 1.0, "strong")]
        result = client.generate(specs, "perception_direct")
        assert result is not None

    def test_ai2thor_triplets_in_perception_records(self):
        AI2THOR_PATH = Path(__file__).parent.parent / "data/raw/ai2thor/claims_all.jsonl"
        if not AI2THOR_PATH.exists():
            pytest.skip("AI2THOR claims not available")
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH, ai2thor_path=AI2THOR_PATH)
        specs = [EvidenceSpec("supports", "ai2thor_simulation", ["perception"], 1.0, "strong")]
        result = client.generate(specs, "perception_direct")
        assert result is not None
        triples = result.get("evidence_triples", [[]])[0]
        assert len(triples) > 0, "Perception records from AI2THOR should carry real triples"

    def test_ai2thor_triplets_flow_through_to_record(self):
        AI2THOR_PATH = Path(__file__).parent.parent / "data/raw/ai2thor/claims_all.jsonl"
        if not AI2THOR_PATH.exists():
            pytest.skip("AI2THOR claims not available")
        from src.core.claims.labels import load_source_trust_registry
        reg = load_source_trust_registry(REGISTRY_PATH)
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH, ai2thor_path=AI2THOR_PATH)
        gen = FictionalClaimGenerator(registry=reg, _client=client)
        # Generate a few records and check that perception ones have triples
        batch = gen.generate_batch(n_records=50)
        perception_recs = [
            r for r in batch
            if r["meta"]["template_type"] == "perception_direct"
        ]
        assert perception_recs, "Expected some perception_direct records in 50-record batch"
        for rec in perception_recs:
            ev_triples = rec["evidence"][0]["triples"]
            assert len(ev_triples) > 0, f"perception_direct record should have triples: {rec['id']}"


# ---------------------------------------------------------------------------
# LLMClient (mocked)
# ---------------------------------------------------------------------------

class TestLLMClient:
    @staticmethod
    def _mock_api(text: str):
        content = MagicMock()
        content.text = text
        msg = MagicMock()
        msg.content = [content]
        api = MagicMock()
        api.messages.create.return_value = msg
        return api

    def test_generate_returns_parsed(self):
        payload = '{"claim": "Test claim.", "evidence_texts": ["Ev A.", "Ev B."]}'
        client = LLMClient(_client=self._mock_api(payload))
        specs = [
            EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong"),
            EvidenceSpec("supports", "apnews_web_text",  ["testimony"], 0.8, "strong"),
        ]
        result = client.generate(specs, "high_trust_supported")
        assert result["claim"] == "Test claim."
        assert len(result["evidence_texts"]) == 2

    def test_api_error_returns_none(self):
        api = MagicMock()
        api.messages.create.side_effect = Exception("API error")
        client = LLMClient(_client=api)
        specs = [EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong")]
        assert client.generate(specs, "test") is None

    def test_invalid_response_returns_none(self):
        client = LLMClient(_client=self._mock_api("Sorry, cannot help."))
        specs = [EvidenceSpec("supports", "reuters_web_text", ["testimony"], 0.8, "strong")]
        assert client.generate(specs, "test") is None


# ---------------------------------------------------------------------------
# FictionalClaimGenerator
# ---------------------------------------------------------------------------

class TestFictionalClaimGenerator:
    def test_generate_batch_count(self, registry):
        client = LocalTextClient()
        gen = FictionalClaimGenerator(registry=registry, _client=client)
        batch = gen.generate_batch(n_records=20)
        assert len(batch) == 20

    def test_shortcut_fraction_above_minimum(self, registry):
        client = LocalTextClient()
        gen = FictionalClaimGenerator(registry=registry, _client=client)
        batch = gen.generate_batch(n_records=100)
        shortcut = sum(1 for r in batch if r["meta"]["is_shortcut_breaking"])
        assert shortcut / len(batch) >= MIN_SHORTCUT_FRACTION

    def test_all_evidence_types_covered(self, registry):
        client = LocalTextClient()
        gen = FictionalClaimGenerator(registry=registry, _client=client)
        batch = gen.generate_batch(n_records=200)
        all_types = {t for r in batch for e in r["evidence"] for t in e["evidence_types"]}
        expected = {EvidenceType.TESTIMONY.value, EvidenceType.PERCEPTION.value,
                    EvidenceType.INFERENCE.value, EvidenceType.COMPARISON_ANALOGY.value,
                    EvidenceType.NON_APPREHENSION.value}
        assert expected <= all_types, f"Missing types: {expected - all_types}"

    def test_defaults_to_local_client_when_no_api_key(self, registry):
        """FictionalClaimGenerator should work without any API key."""
        gen = FictionalClaimGenerator(registry=registry)
        batch = gen.generate_batch(n_records=5)
        assert len(batch) == 5

    def test_generate_with_grounded_client(self, registry):
        client = GroundedClient(seed_pool_path=SEED_POOL_PATH)
        gen = FictionalClaimGenerator(registry=registry, _client=client)
        batch = gen.generate_batch(n_records=10)
        assert len(batch) == 10


# ---------------------------------------------------------------------------
# SyntheticDataValidator
# ---------------------------------------------------------------------------

class TestSyntheticDataValidator:
    def test_passes_with_sufficient_shortcut_fraction(self, registry):
        records = []
        for _ in range(40):
            records.append(_build_record(
                {"claim": "T.", "evidence_texts": ["E."]},
                _TEMPLATES["low_trust_nee"], registry,
            ))
        for _ in range(60):
            records.append(_build_record(
                {"claim": "T.", "evidence_texts": ["E1.", "E2."]},
                _TEMPLATES["high_trust_supported"], registry,
            ))
        val = SyntheticDataValidator(registry=registry)
        report = val.validate_batch(records)
        assert report.shortcut_fraction >= MIN_SHORTCUT_FRACTION
        assert report.ec_mismatch == 0
        assert report.missing_v3_fields == 0

    def test_fails_with_insufficient_shortcut_fraction(self, registry):
        records = [
            _build_record({"claim": "T.", "evidence_texts": ["E1.", "E2."]},
                          _TEMPLATES["high_trust_supported"], registry)
            for _ in range(100)
        ]
        val = SyntheticDataValidator(registry=registry)
        report = val.validate_batch(records)
        assert not report.passes
        assert any("Shortcut fraction" in e for e in report.errors)

    def test_detects_missing_v3_fields(self, registry):
        records = [_build_record({"claim": "T.", "evidence_texts": ["E."]},
                                 _TEMPLATES["low_trust_nee"], registry)]
        records[0]["evidence"][0].pop("source_id")
        val = SyntheticDataValidator(registry=registry)
        report = val.validate_batch(records)
        assert report.missing_v3_fields == 1

    def test_empty_batch_fails(self, registry):
        val = SyntheticDataValidator(registry=registry)
        report = val.validate_batch([])
        assert not report.passes

    def test_summary_contains_key_info(self, registry):
        records = [_build_record({"claim": "T.", "evidence_texts": ["E1.", "E2."]},
                                 _TEMPLATES["high_trust_supported"], registry)]
        val = SyntheticDataValidator(registry=registry)
        report = val.validate_batch(records)
        summary = report.summary()
        assert "Total:" in summary
        assert "Shortcut-breaking:" in summary
        assert "Status:" in summary
