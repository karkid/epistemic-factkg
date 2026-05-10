# ADR-006: Diminishing Returns Formula for Multi-Pramana Confidence

## Status

Accepted

## Context

Some claims are supported by multiple types of evidence simultaneously — for example, an AVeriTeC claim backed by both a direct web document (testimony) and a numeric comparison (comparison_analogy). When multiple Pramana labels apply, we need a principled formula to combine their individual confidence weights into a single `confidence_weight` for the unified record.

Alternatives considered:

| Method | Formula | Problem |
|---|---|---|
| **Max** | `max(wᵢ)` | Ignores all evidence except the strongest; doesn't reward corroboration |
| **Mean** | `mean(wᵢ)` | Adding a weak source dilutes the strong one; unintuitive (adding evidence *reduces* confidence) |
| **Capped sum** | `min(1.0, Σwᵢ)` | Arbitrary ceiling; sum can exceed 1.0 inconsistently |
| **Probabilistic union** | `1 - Π(1 - wᵢ)` | Models "probability at least one source is correct"; principled probabilistic interpretation |

## Decision

Use the **probabilistic union (diminishing returns)** formula:

```
combined = 1 - Π(1 - wᵢ)
```

This models the probability that at least one of the epistemic sources is correct, under the assumption that sources are independent. The formula is implemented in `src/core/claims/labels.py:combine_pramana_weights()`.

**Example calculations:**

```
perception (0.95) + inference (0.55):
  combined = 1 - (1-0.95) × (1-0.55)
           = 1 - 0.05 × 0.45
           = 1 - 0.0225 = 0.9775

testimony (0.80) + comparison_analogy (0.65):
  combined = 1 - (1-0.80) × (1-0.65)
           = 1 - 0.20 × 0.35
           = 1 - 0.07 = 0.93
```

The strongest Pramana always dominates because it already covers most of the probability space. Additional sources add diminishing marginal gain, which matches the intuition that corroboration is valuable but the first strong source carries most of the weight.

## Consequences

**Positive:**
- Combined weight is always ≥ any individual weight — adding evidence never decreases confidence
- The dominant (highest-weight) Pramana drives the combined value; weaker corroborating sources add only marginal gain
- Principled probabilistic interpretation that can be explained to reviewers and cited

**Negative:**
- The independence assumption is an approximation — two web sources (both testimony) are correlated, so the formula will over-estimate combined confidence for same-type evidence pairs
- For most current records, only one Pramana applies, so the formula is only active in edge cases; its value will be most visible with future datasets that have richer multi-source evidence
