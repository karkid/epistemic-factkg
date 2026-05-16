# ADR-017: IS Jitter in Synthetic Data Generation

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-013 (synthetic pipeline), ADR-010 (per-evidence IS)

---

## Context

Synthetic records are generated from fixed templates. Each template specifies a
constant `inference_strength` value (e.g., `low_trust_nee` always uses IS = 0.2).
This means the IS values in synthetic training data cluster at a small set of
discrete points:

```
IS ∈ {0.2, 0.4, 0.6, 0.7, 0.8, 0.9, …}  # only template values appear
```

The ISHead (H2) is trained to regress IS. With clustered targets it learns the
template lookup rather than a continuous IS predictor — it can memorise template
mappings but will fail on out-of-distribution IS values seen in real data
(AVeriTeC, AI2THOR).

---

## Decision

Add Gaussian jitter to each evidence item's IS value during synthetic generation:

```python
is_val = clip(template_IS + N(0, σ=0.05), min=0.1, max=1.0)
```

The jittered `is_val` is used in both the stored `inference_strength` field and in
the EC formula computation — so the EC value in the record remains mathematically
consistent with the actual (jittered) IS.

σ = 0.05 spreads each template cluster by ±0.1 without overlapping adjacent clusters
(which are typically separated by ≥ 0.15).

---

## Alternatives Considered

**A. Wider jitter (σ = 0.1)** — risks overlap between template clusters and makes
IS targets ambiguous for the ISHead. Rejected.

**B. No jitter** — accepted for early iterations; causes IS to be effectively
categorical. Changed after IS RMSE plateaued early in training.

---

## Consequences

- IS targets become continuous, improving ISHead generalisation to real data.
- EC values in synthetic records remain consistent (jitter is applied before EC
  computation, not after).
- The IS distribution in synthetic data broadens; this slightly reduces the
  IS mean across the full training set.
