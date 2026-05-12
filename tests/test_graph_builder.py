"""Tests for ClaimGraphBuilder — node/edge counts, metadata, edge weights (ADR-014)."""

from __future__ import annotations

import pytest

from src.core.gnn.featurizer import Featurizer
from src.core.gnn.graph_builder import ClaimGraphBuilder
from src.core.gnn.types import VERDICT_TO_INT


@pytest.fixture()
def builder():
    return ClaimGraphBuilder(Featurizer())


@pytest.fixture()
def ai2thor_record():
    return {
        "schema_version": "2.0",
        "id": "test-ai2thor-001",
        "claim": "The apple is on the table.",
        "verdict": {"label": "supported"},
        "epistemic": {
            "pramana_primary": "perception",
            "pramana_all": ["perception"],
            "confidence_weight": 0.95,
        },
        "claim_triples": [
            ["entity:Apple", "isOn", "entity:Table"],
            ["entity:Apple", "isClean", "True"],
        ],
        "evidence": [
            {
                "text": "The apple is on the table.",
                "modality": "simulation_state",
                "stance": "supports",
            }
        ],
        "provenance": {"dataset": "ai2thor", "context_id": "FloorPlan1"},
    }


@pytest.fixture()
def averitec_record():
    return {
        "schema_version": "2.0",
        "id": "test-averitec-001",
        "claim": "The president signed the bill.",
        "verdict": {"label": "refuted"},
        "epistemic": {
            "pramana_primary": "testimony",
            "pramana_all": ["testimony"],
            "confidence_weight": 0.80,
        },
        "claim_triples": None,
        "evidence": [
            {
                "text": "The president vetoed the bill.",
                "modality": "web_text",
                "stance": "refutes",
            },
            {
                "text": "No signing ceremony was held.",
                "modality": "web_text",
                "stance": "refutes",
            },
        ],
        "provenance": {"dataset": "averitec", "context_id": "averitec"},
    }


@pytest.fixture()
def nee_record():
    # AVeriTeC NEE: web search found no answer → stance is JSON null (Python None).
    # Distinct from AI2THOR "absent" which is a positive confirmation of physical absence.
    return {
        "schema_version": "2.0",
        "id": "test-nee-001",
        "claim": "Kenya had 500 magistrates in 2020.",
        "verdict": {"label": "not_enough_evidence"},
        "epistemic": {
            "pramana_primary": "non_apprehension",
            "pramana_all": ["non_apprehension"],
            "confidence_weight": 0.75,
        },
        "claim_triples": None,
        "evidence": [
            {
                "text": "How many magistrates were there in Kenya in 2020? No answer could be found.",
                "modality": "unanswerable",
                "stance": None,
            }
        ],
        "provenance": {"dataset": "averitec", "context_id": "averitec"},
    }


@pytest.fixture()
def ai2thor_absence_record():
    # AI2THOR absence-confirmed: simulation found the object absent → stance "absent" → verdict supported.
    # "There is no vase" + absence confirmed = supported (differs from NEE which is unresolved).
    return {
        "schema_version": "2.0",
        "id": "test-absence-001",
        "claim": "There is no vase in this scene.",
        "verdict": {"label": "supported"},
        "epistemic": {
            "pramana_primary": "non_apprehension",
            "pramana_all": ["non_apprehension"],
            "confidence_weight": 0.75,
        },
        "claim_triples": [],
        "evidence": [
            {
                "text": "No sensor evidence found for this object type.",
                "modality": "simulation_state",
                "stance": "absent",
            }
        ],
        "provenance": {"dataset": "ai2thor", "context_id": "FloorPlan2"},
    }


class TestAI2ThorGraph:
    def test_claim_node_shape(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["claim"].x.shape == (1, 384)

    def test_evidence_node_shape(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["evidence"].x.shape == (1, 389)  # 384 + 5 modality

    def test_epistemic_node_shape(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["epistemic"].x.shape == (1, 6)  # 5 pramana + 1 weight

    def test_triple_nodes_present(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["triple"].x.shape[0] == 2  # 2 triples in fixture

    def test_triple_node_shape(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["triple"].x.shape == (2, 384)

    def test_has_epistemic_edge_weight(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        edge_attr = cg.data["claim", "has_epistemic", "epistemic"].edge_attr
        assert edge_attr is not None
        assert abs(edge_attr.item() - 0.95) < 1e-5

    def test_supports_edge_present(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert ("evidence", "supports", "claim") in cg.data.edge_types

    def test_refutes_edge_empty_for_supports_stance(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.data["evidence", "refutes", "claim"].edge_index.shape[1] == 0

    def test_has_triple_edge_present(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert ("claim", "has_triple", "triple") in cg.data.edge_types

    def test_label_correct(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.label == VERDICT_TO_INT["supported"]

    def test_metadata_pramana(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.pramana == "perception"

    def test_metadata_dataset(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        assert cg.dataset == "ai2thor"


class TestAVeriTeCGraph:
    def test_no_triple_nodes(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert cg.data["triple"].x.shape[0] == 0

    def test_evidence_count(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert cg.data["evidence"].x.shape[0] == 2

    def test_refutes_edge_present(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert ("evidence", "refutes", "claim") in cg.data.edge_types

    def test_supports_edge_empty_for_refutes_stance(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert cg.data["evidence", "supports", "claim"].edge_index.shape[1] == 0

    def test_no_has_triple_edge(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert cg.data["claim", "has_triple", "triple"].edge_index.shape[1] == 0

    def test_label_refuted(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        assert cg.label == VERDICT_TO_INT["refuted"]

    def test_confidence_weight_testimony(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        edge_attr = cg.data["claim", "has_epistemic", "epistemic"].edge_attr
        assert abs(edge_attr.item() - 0.80) < 1e-5


class TestNEEGraph:
    """AVeriTeC NEE: stance=null in JSON → no_evidence edge (not absent edge)."""

    def test_no_evidence_edge_present(self, builder, nee_record):
        cg = builder.build(nee_record)
        assert cg.data["evidence", "no_evidence", "claim"].edge_index.shape[1] == 1

    def test_absent_edge_empty_for_nee(self, builder, nee_record):
        cg = builder.build(nee_record)
        assert cg.data["evidence", "absent", "claim"].edge_index.shape[1] == 0

    def test_supports_and_refutes_empty(self, builder, nee_record):
        cg = builder.build(nee_record)
        assert cg.data["evidence", "supports", "claim"].edge_index.shape[1] == 0
        assert cg.data["evidence", "refutes", "claim"].edge_index.shape[1] == 0

    def test_label_nee(self, builder, nee_record):
        cg = builder.build(nee_record)
        assert cg.label == VERDICT_TO_INT["not_enough_evidence"]


class TestAI2ThorAbsenceGraph:
    """AI2THOR absence-confirmed: stance='absent' → absent edge → verdict supported."""

    def test_absent_edge_present(self, builder, ai2thor_absence_record):
        cg = builder.build(ai2thor_absence_record)
        assert cg.data["evidence", "absent", "claim"].edge_index.shape[1] == 1

    def test_no_evidence_edge_empty(self, builder, ai2thor_absence_record):
        cg = builder.build(ai2thor_absence_record)
        assert cg.data["evidence", "no_evidence", "claim"].edge_index.shape[1] == 0

    def test_label_supported(self, builder, ai2thor_absence_record):
        cg = builder.build(ai2thor_absence_record)
        assert cg.label == VERDICT_TO_INT["supported"]


class TestModality:
    def test_simulation_state_modality_index(self, builder, ai2thor_record):
        cg = builder.build(ai2thor_record)
        modality_part = cg.data["evidence"].x[0][384:]  # last 5 dims
        # simulation_state → index 0
        assert modality_part[0].item() == pytest.approx(1.0)
        assert modality_part[1:].sum().item() == pytest.approx(0.0)

    def test_web_text_modality_index(self, builder, averitec_record):
        cg = builder.build(averitec_record)
        modality_part = cg.data["evidence"].x[0][384:]
        # web_text → index 1
        assert modality_part[1].item() == pytest.approx(1.0)
        assert modality_part[0].item() == pytest.approx(0.0)
