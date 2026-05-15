# ADR-022: Shortcut Leakage and Synthetic Data Strategy

## Status

Accepted

## Context

After integrating AI2THOR and AVeriTeC records into the unified dataset, a structural shortcut was identified:

**The problem:** In both AI2THOR and AVeriTeC, `evidence.stance = supports` predicts `verdict = supported` with near-perfect reliability. A GNN trained on this data can achieve high accuracy by learning a trivial stance-lookup rule:

```
if all(e.stance == "supports"): verdict = supported
if all(e.stance == "refutes"):  verdict = refuted
```

This is the **stance→verdict shortcut**. The model does not need to learn:
- Source trustworthiness (ST)
- Inference strength (IS)
- Evidence type weights (EW)
- The EC formula: EC = 1 − (1 − ST)^(EW × IS)

The shortcut is structural, not incidental. It arises because real-world claims with weak evidence are still labelled by annotators based on content — the annotator knows the claim is false, so they label `refuted`, and the evidence they cite is genuinely refuting. Low-trust or hedged evidence almost never appears without a matching verdict.

## Decision

Add a third data source — **synthetic shortcut-breaking records** — where the same stance does NOT imply the same verdict.

The key insight: epistemic reliability must vary independently of stance for the GNN to be forced to learn the EC formula. Specifically:

| Evidence stance | Verdict | Mechanism |
|---|---|---|
| supports | supported | High ST + high IS → EC ≥ 0.75 |
| supports | not_enough_evidence | Low ST or low IS → EC < 0.75 (shortcut-breaking) |
| refutes | refuted | High ST + high IS → EC ≥ 0.75 |
| refutes | not_enough_evidence | Weak refutation → EC < 0.75 (shortcut-breaking) |
| supports + refutes | conflicting_evidence | Both scores ≥ 0.40 (shortcut-breaking) |

**Minimum floor:** ≥ 35% of synthetic records must be shortcut-breaking (stance and verdict point in different directions).

**Implementation:** See ADR-023 for the generation pipeline. See ADR-024 for the grounded generation design.

## Consequences

**Positive:**
- GNN is forced to learn EC-weighted aggregation, not stance lookup
- Shortcut fraction is measurable and enforceable via `SyntheticDataValidator`
- Template-based generation guarantees the EC math is correct by construction — no annotation or LLM judgment is needed to determine the verdict
- All four verdict classes are now represented with controllable frequency

**Negative:**
- Synthetic records are fictional (no real-world grounding for most templates)
- The 35% shortcut floor is heuristic — no theoretical justification for the threshold
- Class imbalance introduced by synthetic data must be monitored (synthetic adds `not_enough_evidence` and `conflicting_evidence` which are underrepresented in the real sources)
