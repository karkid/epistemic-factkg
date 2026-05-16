# ADR-015: Class-Weighted CrossEntropyLoss for Stance and Verdict

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-007 (verdict class reduction), ADR-013 (architecture)

---

## Context

After the first full training run the class distribution on the test set showed severe imbalance:

| Task    | Class               | Share  |
|---------|---------------------|--------|
| Verdict | not_enough_evidence | 11.3%  |
| Stance  | neutral             | ~7%    |

Standard CrossEntropyLoss treats all classes equally. With NEI at 11.3%, the loss is
dominated by supported/refuted gradients and the model learns to ignore NEI, leading
to near-zero NEI recall.

---

## Decision

Apply **inverse-frequency class weights** to both stance and verdict CrossEntropyLoss:

```python
weight_i = N_total / (n_classes × count_i)
```

Weights are computed from training split labels at the start of each training run and
passed to `nn.CrossEntropyLoss(weight=...)`.

Both stance and verdict weights are computed independently from their respective label
distributions. A `--no-class-weights` flag disables both for ablation.

---

## Alternatives Considered

**A. Oversample minority classes** — creates duplicate graphs, risks memorisation of
rare templates. Rejected in favour of loss weighting which is zero-cost and reversible.

**B. Address imbalance at the data source (NEI-heavy synthetic)** — implemented
in parallel (ADR-018). Both approaches are complementary: the data fix is the primary
lever; loss weighting is a safety net if the rebuilt distribution is still skewed.

**C. Focal loss** — downweights easy examples rather than rare classes. Would require
a separate γ hyperparameter. Not implemented because inverse-frequency weighting is
sufficient and interpretable.

---

## Consequences

- NEI recall improves; macro F1 is a more informative headline metric than accuracy.
- The effective learning rate for rare classes increases — must monitor for
  oscillation on small minority batches.
- Per-class breakdown is now reported in `eval_summary.md` to make weighting effects
  visible.
