# ADR-014: Learned Verdict Thresholds via VerdictHead

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-001 (EC formula), ADR-013 (neuro-symbolic architecture)

---

## Context

The symbolic aggregation layer computes `support_score` and `refute_score` per claim
using the EC formula (ADR-001). In the initial V1 architecture (ADR-013), the final
verdict was determined by hard-coded thresholds:

```
support ≥ 0.75 and refute < 0.40  →  supported
refute  ≥ 0.75 and support < 0.40 →  refuted
both    ≥ 0.40                     →  conflicting_evidence
else                               →  not_enough_evidence
```

Evaluation on the held-out test set showed a large per-source gap:

| Source    | Verdict accuracy |
|-----------|-----------------|
| AI2THOR   | 91.1%           |
| Synthetic | 96.9%           |
| AVeriTeC  | 24.8%           |

The root cause is an **annotation–formula mismatch**:

- AI2THOR and synthetic verdicts are **derived by the EC formula** — the symbolic
  scores naturally reproduce the ground-truth verdicts.
- AVeriTeC verdicts are **human-annotated** by fact-checkers who did not apply
  EC-formula thresholds. A claim annotated "supported" may have evidence from
  mixed-trust sources whose aggregated EC scores fall below 0.75, causing the
  symbolic layer to output `not_enough_evidence` instead.

The hard-coded thresholds (0.75 / 0.40) were chosen for the EC formula's semantics,
not for human annotation alignment.

---

## Decision

Replace the hard-coded threshold block with a learned `VerdictHead`:

```python
verdict_logits = Linear(2 → 3)([support_score, refute_score])
```

Trained with `CrossEntropyLoss` against claim-level verdict labels from all sources.
The EC formula and `SymbolicAggregator` are **unchanged** — they still compute
interpretable `support_score` and `refute_score`. The VerdictHead learns where the
decision boundaries sit for each dataset's annotation style.

### Training objective

```
total_loss = stance_CE  +  λ₁ × IS_MSE  +  λ₂ × verdict_CE
```

- `stance_CE`  (H1) — per-evidence, unchanged
- `IS_MSE`     (H2) — per-evidence, unchanged
- `verdict_CE`      — per-claim, new; supervised by annotated claim labels

Default weights: λ₁ = 0.5, λ₂ = 1.0.

### Differentiable soft aggregation

To propagate verdict loss gradients back through H1 and H2, the forward pass uses
**soft symbolic scores** during training:

```
soft_support = 1 − ∏(1 − EC_i × p_support_i)
soft_refute  = 1 − ∏(1 − EC_i × p_refute_i)
```

where `p_support_i` and `p_refute_i` are softmax probabilities from H1 (not argmax).
This keeps the EC formula structure intact while making it differentiable.

At inference, hard argmax is used for interpretable per-evidence stance, and the
VerdictHead maps the resulting scores to a verdict string.

---

## Alternatives Considered

### A. Per-evidence stance fix (rejected)

Re-assign AVeriTeC evidence stance based on source trust (low-trust sources → neutral).
Rejected because:
- **Circular**: uses the source trust registry to assign labels that feed back into
  training, which uses the same registry in symbolic aggregation.
- **Conflates stance and reliability**: a low-trust source can still clearly support
  or refute a claim. Stance (what the evidence says) and trustworthiness (how much
  to weight it) are independent properties.
- Labels remain claim-level approximations, not genuine per-evidence annotations.

### B. Dataset-conditioned verdict head (rejected)

Separate verdict heads for EC-derived datasets (AI2THOR, synthetic) vs.
human-annotated (AVeriTeC). Rejected because it breaks the unified architecture
and adds inference-time dataset routing logic.

### C. Accept the mismatch (rejected for training; retained as ablation)

Report the 25% AVeriTeC accuracy as a finding — EC formula scores diverge from
human annotation thresholds. Retained as an ablation: evaluating with hard thresholds
vs. VerdictHead quantifies how much annotation-style calibration matters.

---

## Consequences

**Positive:**
- AVeriTeC verdict accuracy is expected to improve substantially — the VerdictHead
  learns that lower symbolic scores can still correspond to "supported" in
  human-annotated data.
- EC formula remains interpretable and unchanged; `support_score` and `refute_score`
  are still reported as intermediate outputs.
- Only 6 additional parameters (2 weights + 1 bias × 3 classes).
- End-to-end differentiable: verdict loss provides additional gradient signal to
  both H1 (through soft stance probabilities) and H2 (through soft IS scalars).

**Trade-offs:**
- AVeriTeC verdict supervision uses claim-level labels applied to all evidence
  items uniformly (no per-evidence stance ground truth exists in AVeriTeC).
  This is acknowledged as a data limitation.
- The verdict loss weight λ₂ is a hyperparameter. The ablation in the paper
  should include λ₂ = 0 (hard thresholds only) to isolate the VerdictHead effect.

---

## Ablation Protocol

To isolate the VerdictHead contribution, report two evaluation runs:

| Configuration | `verdict_loss_weight` | Verdict layer |
|---|---|---|
| Baseline (ADR-013) | 0.0 | hard thresholds |
| +VerdictHead (this ADR) | 1.0 | learned Linear(2→3) |

Expected finding: +VerdictHead improves AVeriTeC accuracy with negligible change
to AI2THOR and synthetic (whose EC scores already align with ground truth).
