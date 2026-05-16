# ADR-022: IS Gradient Detach — Decoupling IS from Verdict Gradient Path

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-013 (architecture), ADR-014 (VerdictHead), ADR-016 (baseline ablation)

---

## Context

In EpistemicHGNN the verdict loss backpropagates through the EC formula into the
ISHead:

```
verdict_CE → VerdictHead → scores → EC formula → is_pred → ISHead → encoder
```

This creates a gradient conflict: the IS regression loss pulls IS toward the
ground-truth IS values (from the rubric), while the verdict loss pulls IS toward
whatever values maximise verdict accuracy through the EC formula. These objectives
are not aligned when AVeriTeC labels disagree with EC semantics.

**Evidence of the conflict:** IS RMSE for EpistemicHGNN was 0.234 vs 0.100 for
BaselineHGNN (which has no verdict→IS gradient path), despite both models sharing
the same ISHead architecture. The baseline's IS regression had a single clean
supervisor; v1-hgnn's had two competing ones.

---

## Decision

Detach the IS tensor before it enters the EC formula in `_soft_verdict_logits`:

```python
verdict_logits = self._soft_verdict_logits(data, stance_logits, is_pred.detach())
```

`is_pred` (with gradients) is returned in the forward dict so IS regression loss
still flows through it cleanly. `is_pred.detach()` is the copy fed to the EC
formula — EC backpropagation stops here.

Gradient paths after detach:

| Loss           | Path to encoder                                |
|----------------|------------------------------------------------|
| IS regression  | is_pred → ISHead → encoder  ✓ (clean)         |
| Verdict        | scores → EC(fixed IS) → stance_probs → encoder ✓ |
| Stance         | stance_logits → StanceHead → encoder  ✓       |

---

## Alternatives Considered

**A. Reduce verdict loss weight λ₂** — reduces gradient magnitude but does not
eliminate the conflict. Gradient directions still oppose. Rejected as a partial fix.

**B. Two-stage training** — Stage 1 trains IS only, Stage 2 trains verdict with IS
frozen. Operationally complex; requires a training loop change. The detach achieves
the same decoupling with a single line.

**C. Keep gradient coupling** — the original design. Found task-optimal IS values
(not ground-truth IS) that maximised verdict through EC. Rejected because:
  (1) IS RMSE 0.234 is too high for interpretable epistemic claims;
  (2) the improved IS becomes a bottleneck when passed through a 2D VerdictHead
  (see ADR-023 for the resolution).

---

## Consequences

- IS RMSE improves from ~0.23 to ~0.12; IS Pearson r from 0.68 to 0.87.
- IS prediction quality matches the baseline's clean IS regression.
- Verdict accuracy with pure VerdictHead (v1-hgnn) drops because VerdictHead
  receives only 2D EC scores without the claim embedding — the encoder no longer
  optimises for verdict via the IS pathway. This motivates ADR-023 (HybridHGNN).
- The detach is retained in HybridHGNN (v2-hgnn) where the claim embedding
  provides a direct, rich gradient path to the encoder for verdict supervision.
