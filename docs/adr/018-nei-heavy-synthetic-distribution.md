# ADR-018: NEI-Heavy Synthetic Template Distribution

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-006 (dataset strategy), ADR-007 (verdict classes), ADR-012 (synthetic strategy)

---

## Context

After the first full pipeline run, the verdict distribution in the training set was:

| Verdict              | Share  |
|----------------------|--------|
| refuted              | 51.0%  |
| supported            | 37.7%  |
| not_enough_evidence  | 11.3%  |

NEI at 11.3% is too low for the model to learn the Anupalabdhi (non-apprehension)
signal that NEI claims carry. NEI is the most epistemically distinctive class —
it maps directly to the absence of sufficient evidence, not a positive stance.

The imbalance originates in the synthetic generator: the original template
distribution allocated only ~8% to NEI-producing templates.

---

## Decision

Redesign the synthetic template distribution to allocate ~55% of records to
NEI-producing templates:

```yaml
# NEI-heavy templates (~55%)
low_trust_nee:                0.11
low_trust_refuted_nee:        0.11
weak_vs_weak_nee:             0.12
inference_nee:                0.11
non_apprehension_weak_nee:    0.10

# Supported templates (~21%)
high_trust_supported:         0.05
strong_support_weak_refute:   0.07
corroborating_3:              0.05
perception_direct:            0.05
comparison_supported:         0.04
non_apprehension_absent:      0.02  # absence supports presence-claim

# Refuted templates (~24%)
high_trust_refuted:           0.06
weak_support_strong_refute:   0.07
non_apprehension_refuted:     0.04
```

The `conflicting_evidence` template is removed entirely (see ADR-007).

---

## Alternatives Considered

**A. Keep balanced distribution, rely only on loss weighting (ADR-015)** — loss
weighting is a training-time correction, not a data fix. If NEI examples are too
rare the model sees very few NEI gradient steps per epoch regardless of weights.
Both fixes are applied as complementary layers.

**B. Increase total synthetic record count** — raising from 1000 to 2500 records
provides more absolute NEI examples even at the old distribution. Combined with
the NEI-heavy distribution this was the implemented approach.

---

## Consequences

- Expected NEI share in full training set rises from ~11% toward ~24% after rebuild.
- Synthetic refuted share drops proportionally; AI2THOR and AVeriTeC provide the
  balance of refuted examples.
- The NEI templates cover all three Pramana-level NEI sources: low source trust,
  weak inference strength, and non-apprehension (absent evidence).
