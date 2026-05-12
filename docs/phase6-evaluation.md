# Phase 6: Evaluation

**Status:** Planned  
**Date:** 2026-05-12  
**ADR:** 017  
**Prerequisite:** Phase 5 complete — all 4 checkpoint directories exist

---

## Overview

Phase 6 evaluates all 4 trained models on the held-out test set (N=517) and produces
the metrics needed to answer the research question:

> Does Pramana-aware epistemic structure improve fact verification compared to text alone?

The test set is used **once**, after all Phase 5 training is complete. No hyperparameter
adjustments are made after seeing test results.

See ADR-017 for the full evaluation protocol.

---

## Evaluation Script

**Script:** `src/cli/evaluate_gnn.py`

```
Arguments:
  --checkpoint PATH    best_model.pt for a specific run
  --dataset PATH       out/graphs/graph_dataset.pt
  --jsonl PATH         out/training/epistemic_factkg_training.jsonl
  --splits-dir PATH    directory containing test_indices.json
  --no-stance-edges    zero stance back-edges (must match training flags)
  --no-epistemic       zero has_epistemic edge (must match training flags)
  --output PATH        directory to write result JSON files
```

The `--no-stance-edges` / `--no-epistemic` flags must mirror the training flags for the
same run. If Run B was trained with `--no-stance-edges`, evaluation must also pass
`--no-stance-edges` so the graph is processed identically at inference time.

---

## Metrics Module

**Module:** `src/core/gnn/metrics.py`

Pure functions — no I/O, fully unit-testable:

```python
compute_accuracy(preds: Tensor, labels: Tensor) -> float
    # fraction of correct predictions

compute_macro_f1(preds: Tensor, labels: Tensor, n_classes: int) -> float
    # unweighted average of per-class F1 scores

compute_confusion_matrix(preds: Tensor, labels: Tensor, n_classes: int) -> list[list[int]]
    # rows = true class, cols = predicted class

compute_ece(logits: Tensor, labels: Tensor, n_bins: int = 10) -> float
    # expected calibration error — bin by max(softmax), measure |confidence - accuracy|

compute_per_group_accuracy(
    preds: Tensor, labels: Tensor, groups: list[str]
) -> dict[str, dict[str, float | int]]
    # groups is a list of string labels aligned with preds/labels
    # returns {group_name: {"accuracy": float, "support": int}}
```

---

## Output Files (per run)

Each run's results are written to `out/results/<run-name>/`:

| File | Contents |
|------|----------|
| `metrics.json` | `accuracy`, `macro_f1`, `weighted_f1`, `ece`, `per_class` |
| `confusion_matrix.json` | 3×3 integer matrix, rows=true, cols=predicted |
| `per_pramana.json` | `{pramana_type: {accuracy, macro_f1, support}}` |
| `per_source.json` | `{ai2thor: {accuracy, support}, averitec: {accuracy, support}}` |

`metrics.json` structure:
```json
{
  "accuracy": 0.72,
  "macro_f1": 0.58,
  "weighted_f1": 0.71,
  "ece": 0.08,
  "per_class": {
    "supported":           {"precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 202},
    "refuted":             {"precision": 0.78, "recall": 0.82, "f1": 0.80, "support": 316},
    "not_enough_evidence": {"precision": 0.55, "recall": 0.40, "f1": 0.46, "support": 34}
  }
}
```

---

## Accessing Per-Pramana and Per-Source Labels

The PyG `.pt` dataset does not store `pramana` or `dataset` strings (excluded to avoid
PyG slice issues with non-tensor attributes). Access them by re-loading the JSONL:

```python
import json
from pathlib import Path

records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
test_indices = json.loads(Path(splits_dir / "test_indices.json").read_text())["indices"]

pramana_labels = [
    records[i]["epistemic"]["pramana_primary"] for i in test_indices
]
source_labels = [
    records[i]["provenance"]["dataset"] for i in test_indices
]
```

The DataLoader must use `shuffle=False` so predictions are aligned with the JSONL ordering.

---

## Running Phase 6

```bash
# Evaluate all 4 runs
just evaluate

# Or individually:
uv run python src/cli/evaluate_gnn.py \
    --checkpoint out/checkpoints/best_model.pt \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --splits-dir out/splits \
    --output out/results/full/

uv run python src/cli/evaluate_gnn.py \
    --checkpoint out/checkpoints/no-stance/best_model.pt \
    --no-stance-edges \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --splits-dir out/splits \
    --output out/results/no-stance/

uv run python src/cli/evaluate_gnn.py \
    --checkpoint out/checkpoints/no-stance-no-epistemic/best_model.pt \
    --no-stance-edges --no-epistemic \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --splits-dir out/splits \
    --output out/results/no-stance-no-epistemic/

uv run python src/cli/evaluate_gnn.py \
    --checkpoint out/checkpoints/pathway-b/best_model.pt \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --splits-dir out/splits \
    --output out/results/pathway-b/
```

---

## Expected Output: Ablation Comparison Table

```
Run                        | Macro F1 | Accuracy | ECE  | NEE F1
---------------------------|----------|----------|------|--------
C  (full graph)            |  ~1.00   |  ~1.00   |  —   |  ~1.00
B  (no-stance + epistemic) |    ?     |    ?     |  ?   |    ?
A  (no-stance, text only)  |    ?     |    ?     |  ?   |    ?
D  (Pathway B)             |  ~1.00   |  ~1.00   |  —   |  ~1.00
```

**Hypothesis supported if:** Run B macro F1 − Run A macro F1 > 5pp (see ADR-017).

---

## Tests

`tests/test_metrics.py` (new):

| Test | What it verifies |
|------|-----------------|
| `test_accuracy_perfect` | All-correct predictions → 1.0 |
| `test_accuracy_half` | 50/50 → 0.5 |
| `test_macro_f1_known` | Compare against sklearn reference values |
| `test_confusion_matrix_shape` | Returns n_classes × n_classes |
| `test_ece_perfect_calibration` | All-confident-correct predictions → ECE ≈ 0 |
| `test_ece_worst_case` | Always-confident-wrong predictions → ECE ≈ 1 |
| `test_per_group_accuracy_two_groups` | Exact per-group values with known inputs |
| `test_per_group_support_counts` | Support counts sum to total |

---

## Results Layout After Phase 6

```
out/results/
├── full/
│   ├── metrics.json
│   ├── confusion_matrix.json
│   ├── per_pramana.json
│   └── per_source.json
├── no-stance/
│   └── ...
├── no-stance-no-epistemic/
│   └── ...
└── pathway-b/
    └── ...
```

Fill sections 3 and 4 of `docs/conclusions.md` once `out/results/` is populated.

---

## Stretch: Attention Weight Extraction

GATConv supports extracting per-edge attention weights via `return_attention_weights=True`.
This would answer: "which evidence nodes did the model attend to when predicting `refuted`?"

Implementation requires adding an `explain_mode: bool` parameter to
`EpistemicHGNN.forward()`. Implement only if Runs A/B produce meaningfully different
results that warrant explanation. This is a Phase 6 stretch goal, not a gate.
