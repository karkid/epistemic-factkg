# ADR-008: Evaluation Protocol

## Status

Accepted

## Context

After Phase 5 produces 4 trained models (Runs A, B, C, D), Phase 6 evaluates them on the
held-out test set to answer the research question. The evaluation design must be decided
before running to prevent post-hoc metric selection, which would inflate the apparent
significance of results.

The test set (`out/splits/test_indices.json`, N=517) has not been used during training or
checkpoint selection. It is evaluated **once**, after all Phase 5 training is complete.

## Decision

### Primary Metric: Macro F1

Macro F1 is the primary metric, not accuracy.

**Rationale:** The test set is imbalanced — refuted 61%, supported 39%, NEE 6.6%. A model
that always predicts "refuted" achieves 61% accuracy but macro F1 ≈ 0.25. Macro F1 computes
F1 for each class independently and averages, weighting each class equally regardless of
support. This makes the metric sensitive to performance on the rare `not_enough_evidence`
class, which is central to the `non_apprehension` Pramana research contribution (ADR-001).

Accuracy is reported alongside macro F1 for interpretability and comparison with prior work.

### Full Metric Set (per run)

| Metric | Type | Notes |
|--------|------|-------|
| Macro F1 | Primary | Equal weight per class |
| Accuracy | Secondary | Raw fraction correct |
| Weighted F1 | Secondary | Weighted by class support — for comparison with prior work |
| Per-class P/R/F1 | Diagnostic | 3 rows: supported, refuted, not_enough_evidence |
| Confusion matrix | Diagnostic | 3×3 — rows=true, cols=predicted |
| ECE | Calibration | Meaningful for Runs A and B only (see below) |

### Calibration: Expected Calibration Error (ECE)

ECE measures whether softmax confidence matches empirical accuracy. Formula:

```
ECE = Σ_b (|B_b| / N) * |avg_confidence(B_b) − avg_accuracy(B_b)|
```

Where bins `B_b` partition [0, 1] into 10 equal intervals over `max(softmax(logits))`.

**ECE is only reported for Runs A and B** (stance removed). For Runs C and D (stance
present), the routing shortcut produces near-certain softmax outputs — ECE would be
trivially low even though the model has learned nothing about epistemic structure. A
well-calibrated Run B model has ECE < 0.05.

### Per-Pramana Breakdown

For each of the 5 Pramana types, report accuracy and support count on test records.

| Pramana | Expected behaviour (if hypothesis supported) |
|---------|---------------------------------------------|
| `non_apprehension` | High NEE recall — Pramana type directly encodes not_enough_evidence |
| `perception` | High supported recall — AI2THOR simulation confirms presence/absence |
| `testimony` | Moderate — web evidence can go either way |
| `inference` | Moderate — derived conclusions are ambiguous |
| `comparison_analogy` | Variable — thin training coverage |

Accessing pramana metadata: re-load `out/training/epistemic_factkg_training.jsonl` and
extract `records[i]["epistemic"]["pramana_primary"]` for each test index `i`. The test
DataLoader must use `shuffle=False` to maintain alignment with the JSONL ordering.

### Per-Source Breakdown

Accuracy split by `provenance.dataset` (values: `ai2thor`, `averitec`). Accessed via
`records[i]["provenance"]["dataset"]` — same JSONL re-loading approach.

This breakdown tests whether the model generalises across sources or learns source-specific
patterns (e.g., all AI2THOR claims are supported or absent, all AVeriTeC may be refuted).

### Test-Set Discipline

1. Test indices are loaded from `out/splits/test_indices.json` — same file used for all runs
2. No hyperparameter decisions are made after seeing test results
3. Checkpoint selection is based on **validation set accuracy only**
4. Phase 6 reports all metrics for all runs simultaneously — no cherry-picking

### What "Hypothesis Supported" Means

The epistemic hypothesis is supported if **both** conditions hold:

**Condition 1:** Run B macro F1 > Run A macro F1 by > 5pp on the test set

This shows the EpistemicNode (Pramana type + confidence weight) adds independent predictive
signal beyond the claim text and evidence embeddings.

**Condition 2:** Per-Pramana accuracy is not uniform across Pramana types in Run B

If condition 1 holds but all 5 Pramana types achieve equal accuracy, the epistemic node
may be acting as a verdict proxy (e.g., `non_apprehension` ≈ NEE verdict) rather than
providing structural epistemic reasoning. Meaningful variation across Pramana types — with
`non_apprehension` and `perception` outperforming `testimony` and `inference` — is evidence
of Pramana-specific structure contributing to predictions.

**Negative results are still publishable.** If Run B ≈ Run A, this shows that the
heuristic Pramana labels from ADR-001 do not add signal beyond what the sentence
embeddings already capture — a meaningful finding about the limits of heuristic epistemic
annotation at this dataset scale.

## Consequences

**Evaluation script design (`src/cli/evaluate_gnn.py`):**
- Must accept `--checkpoint`, `--dataset`, `--jsonl`, `--splits-dir`, `--output`
- Must accept `--no-stance-edges` and `--no-epistemic` flags matching the training flags,
  so graphs are processed identically at inference time
- Must write one JSON file per metric type to `--output` directory

**New module `src/core/gnn/metrics.py`:**
Pure functions (no I/O, fully unit-testable):
- `compute_accuracy(preds, labels) → float`
- `compute_macro_f1(preds, labels, n_classes) → float`
- `compute_confusion_matrix(preds, labels, n_classes) → list[list[int]]`
- `compute_ece(logits, labels, n_bins=10) → float`
- `compute_per_group_accuracy(preds, labels, groups) → dict[str, dict]`

**Reporting format:** Each run's results are written to `out/results/<run-name>/`.
The final comparison table across all runs is assembled manually or via a notebook.

**Attention weight extraction (stretch target):**
GATConv supports `return_attention_weights=True` in its forward call. Extracting per-edge
attention weights would allow Phase 6 to answer: "which evidence items did the model attend
to when predicting `refuted`?" This is deferred — implement only if Runs A/B produce
meaningfully different results worth explaining.
