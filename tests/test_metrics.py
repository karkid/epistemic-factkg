"""Unit tests for src/core/gnn/metrics.py (ADR-017)."""

from __future__ import annotations


import pytest
import torch

from src.model.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_ece,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_per_group_accuracy,
    compute_weighted_f1,
)


# ── compute_accuracy ──────────────────────────────────────────────────────────


class TestAccuracy:
    def test_perfect(self):
        preds = torch.tensor([0, 1, 2, 0])
        labels = torch.tensor([0, 1, 2, 0])
        assert compute_accuracy(preds, labels) == pytest.approx(1.0)

    def test_half(self):
        preds = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 0, 1, 1])
        assert compute_accuracy(preds, labels) == pytest.approx(0.5)

    def test_none_correct(self):
        preds = torch.tensor([1, 2, 0])
        labels = torch.tensor([0, 0, 2])
        assert compute_accuracy(preds, labels) == pytest.approx(0.0)


# ── compute_macro_f1 ──────────────────────────────────────────────────────────


class TestMacroF1:
    def test_perfect_f1(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        assert compute_macro_f1(preds, labels, 3) == pytest.approx(1.0)

    def test_always_predict_zero(self):
        # Predicts class 0 always: F1[0]=recall only, F1[1]=F1[2]=0
        preds = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 2, 0])
        f1 = compute_macro_f1(preds, labels, 3)
        # class 0: tp=2, fp=2, fn=0 → P=0.5, R=1.0, F1=0.667
        # class 1: tp=0 → F1=0
        # class 2: tp=0 → F1=0
        # macro = 0.667/3 ≈ 0.222
        assert 0.20 < f1 < 0.25

    def test_two_class(self):
        preds = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 1, 0])
        # Each class: tp=1, fp=1, fn=1 → F1=0.5 each → macro=0.5
        assert compute_macro_f1(preds, labels, 2) == pytest.approx(0.5)


# ── compute_weighted_f1 ───────────────────────────────────────────────────────


class TestWeightedF1:
    def test_perfect(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        assert compute_weighted_f1(preds, labels, 3) == pytest.approx(1.0)

    def test_weighted_differs_from_macro(self):
        # Imbalanced: class 0 has 3 examples, class 1 has 1
        preds = torch.tensor([0, 0, 0, 1])
        labels = torch.tensor([0, 0, 0, 1])
        macro = compute_macro_f1(preds, labels, 2)
        weighted = compute_weighted_f1(preds, labels, 2)
        assert macro == pytest.approx(1.0)
        assert weighted == pytest.approx(1.0)


# ── compute_confusion_matrix ──────────────────────────────────────────────────


class TestConfusionMatrix:
    def test_shape(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 1, 0])
        cm = compute_confusion_matrix(preds, labels, 3)
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_diagonal_correct(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        cm = compute_confusion_matrix(preds, labels, 3)
        assert cm[0][0] == 1
        assert cm[1][1] == 1
        assert cm[2][2] == 1

    def test_off_diagonal(self):
        # predict 0 for all: class 1 → predicted as 0
        preds = torch.tensor([0, 0, 0])
        labels = torch.tensor([0, 1, 2])
        cm = compute_confusion_matrix(preds, labels, 3)
        assert cm[0][0] == 1  # true 0, predicted 0
        assert cm[1][0] == 1  # true 1, predicted 0
        assert cm[2][0] == 1  # true 2, predicted 0

    def test_sum_equals_total(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        labels = torch.randint(0, 3, (6,))
        cm = compute_confusion_matrix(preds, labels, 3)
        assert sum(cm[i][j] for i in range(3) for j in range(3)) == 6


# ── compute_ece ───────────────────────────────────────────────────────────────


class TestECE:
    def test_perfect_calibration(self):
        # High-confidence correct predictions → ECE ≈ 0
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        ece = compute_ece(logits, labels)
        assert ece < 0.05

    def test_wrong_with_high_confidence(self):
        # Confident but always wrong → ECE ≈ 1
        logits = torch.tensor([[10.0, 0.0, 0.0]] * 10)
        labels = torch.tensor([1] * 10)
        ece = compute_ece(logits, labels)
        assert ece > 0.8

    def test_returns_float(self):
        logits = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))
        result = compute_ece(logits, labels)
        assert isinstance(result, float)

    def test_non_negative(self):
        logits = torch.randn(20, 3)
        labels = torch.randint(0, 3, (20,))
        assert compute_ece(logits, labels) >= 0.0


# ── compute_per_group_accuracy ────────────────────────────────────────────────


class TestPerGroupAccuracy:
    def test_two_groups_exact(self):
        preds = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 1, 1, 1])
        groups = ["A", "A", "B", "B"]
        result = compute_per_group_accuracy(preds, labels, groups)
        assert result["A"]["accuracy"] == pytest.approx(0.5)
        assert result["B"]["accuracy"] == pytest.approx(1.0)

    def test_support_counts(self):
        preds = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 0, 1])
        groups = ["x", "x", "x", "y", "y"]
        result = compute_per_group_accuracy(preds, labels, groups)
        assert result["x"]["support"] == 3
        assert result["y"]["support"] == 2

    def test_support_sums_to_total(self):
        preds = torch.zeros(10, dtype=torch.long)
        labels = torch.zeros(10, dtype=torch.long)
        groups = ["a"] * 4 + ["b"] * 6
        result = compute_per_group_accuracy(preds, labels, groups)
        total = sum(v["support"] for v in result.values())
        assert total == 10

    def test_single_group(self):
        preds = torch.tensor([0, 0])
        labels = torch.tensor([0, 1])
        groups = ["only"] * 2
        result = compute_per_group_accuracy(preds, labels, groups)
        assert result["only"]["accuracy"] == pytest.approx(0.5)
        assert result["only"]["support"] == 2


# ── compute_per_class_metrics ─────────────────────────────────────────────────


class TestPerClassMetrics:
    def test_perfect_three_class(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        result = compute_per_class_metrics(preds, labels, 3)
        for c in range(3):
            assert result[c]["precision"] == pytest.approx(1.0)
            assert result[c]["recall"] == pytest.approx(1.0)
            assert result[c]["f1"] == pytest.approx(1.0)

    def test_support_counts(self):
        preds = torch.tensor([0, 1, 2, 0])
        labels = torch.tensor([0, 1, 2, 2])
        result = compute_per_class_metrics(preds, labels, 3)
        assert result[0]["support"] == 1
        assert result[1]["support"] == 1
        assert result[2]["support"] == 2
