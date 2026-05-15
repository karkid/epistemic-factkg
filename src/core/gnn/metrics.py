"""Pure metric functions for EpistemicHGNN evaluation (ADR-013)."""

from __future__ import annotations

import torch


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct predictions."""
    return (preds == labels).float().mean().item()


def compute_macro_f1(
    preds: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> float:
    """Unweighted average of per-class F1 scores."""
    f1_scores: list[float] = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)
    return sum(f1_scores) / n_classes


def compute_weighted_f1(
    preds: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> float:
    """F1 averaged weighted by class support (number of true instances per class)."""
    total = labels.size(0)
    weighted_sum = 0.0
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = (labels == c).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        weighted_sum += f1 * support
    return weighted_sum / total if total > 0 else 0.0


def compute_per_class_metrics(
    preds: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> dict[int, dict[str, float]]:
    """Precision, recall, F1, and support for each class."""
    result: dict[int, dict[str, float]] = {}
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = int((labels == c).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return result


def compute_confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor, n_classes: int
) -> list[list[int]]:
    """Return n_classes×n_classes confusion matrix (rows=true, cols=predicted)."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true][pred] += 1
    return matrix


def compute_ece(
    logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> float:
    """Expected Calibration Error — bin by max softmax confidence (ADR-017)."""
    probs = torch.softmax(logits, dim=-1)
    confidences, predicted = probs.max(dim=-1)
    correct = predicted.eq(labels).float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = labels.size(0)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        # include upper boundary in last bin
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        count = mask.sum().item()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean().item()
        avg_acc = correct[mask].mean().item()
        ece += (count / total) * abs(avg_conf - avg_acc)

    return round(ece, 4)


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Root-mean-square error between predicted and ground-truth IS scalars."""
    return round(torch.sqrt(torch.mean((preds.view(-1) - targets.view(-1)) ** 2)).item(), 4)


def compute_pearson_r(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Pearson correlation coefficient between predicted and ground-truth IS scalars."""
    p = preds.view(-1).float()
    t = targets.view(-1).float()
    if p.numel() < 2:
        return 0.0
    p_c = p - p.mean()
    t_c = t - t.mean()
    num = (p_c * t_c).sum()
    den = torch.sqrt((p_c ** 2).sum() * (t_c ** 2).sum())
    if den.item() == 0:
        return 0.0
    return round((num / den).item(), 4)


def compute_per_group_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
) -> dict[str, dict[str, float | int]]:
    """Per-group accuracy and support count.

    Args:
        preds:  Predicted class indices [N].
        labels: True class indices [N].
        groups: String group label for each sample (length N).

    Returns:
        {group_name: {"accuracy": float, "support": int}}
    """
    result: dict[str, dict[str, float | int]] = {}
    unique_groups = sorted(set(groups))
    for grp in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        grp_preds = preds[idx_t]
        grp_labels = labels[idx_t]
        accuracy = (grp_preds == grp_labels).float().mean().item()
        result[grp] = {"accuracy": round(accuracy, 4), "support": len(idx)}
    return result
