# ADR-001: Pramana Epistemic Framework

**Status:** Accepted
**Merges:** ADR-001 (Pramana categories), ADR-005 (non_apprehension vs NEE), ADR-006 (combination formula), ADR-008 (weight values)

---

## Context

Fact verification systems typically treat all evidence as epistemically equivalent — a sensor reading, a web article, and a multi-hop inference carry the same weight. We needed a principled way to assign reliability priors to evidence items from sources with fundamentally different epistemic properties (AI2THOR simulation vs. AVeriTeC web text).

Alternatives considered: uniform weights (no epistemic signal), empirically estimated weights (no calibration data), domain-expert annotation (too slow). Chosen: theoretically motivated weights from the Pramana system.

---

## Decision

### 1. Six Pramana categories

Adopt the six-category Pramana taxonomy as the `evidence_types` label set. Each category encodes how knowledge was obtained:

| Pramana | Label | Primary use | Weight |
|---|---|---|---|
| Pratyakṣa | `perception` | AI2THOR — sensor-confirmed fact | **0.95** |
| Śabda | `testimony` | Web text, PDFs, cited sources | **0.80** |
| Anupalabdhi | `non_apprehension` | AI2THOR — sensor-confirmed absence | **0.75** |
| Upamāna | `comparison_analogy` | Numeric or analogy-based claims | **0.65** |
| Anumāna | `inference` | Multi-source or multi-hop reasoning | **0.55** |
| Arthāpatti | `postulation_derivation` | Hypothetical derivation (excluded from training — ADR-005) | **0.40** |

Weight rationale: `perception` is near-1.0 because AI2THOR is a closed-world ground truth. `testimony` reflects web source reliability. Weights decrease as reasoning steps increase. No weight is 0.0 or 1.0 — even the best source is not infallible. Values are defined in `src/core/claims/labels.py:CONFIDENCE_WEIGHTS`.

### 2. Multi-type combination formula

When an evidence item has multiple `evidence_types`, combine weights via probabilistic union (diminishing returns):

```
EW = 1 - Π(1 - wᵢ)
```

This models "probability at least one source is correct" under source independence. The strongest type always dominates; additional types add diminishing marginal gain. Implemented in `src/core/claims/labels.py:combine_pramana_weights()`.

Examples:
- `["perception", "inference"]` → `1 - (1-0.95)(1-0.55)` = **0.9775**
- `["testimony", "comparison_analogy"]` → `1 - (1-0.80)(1-0.65)` = **0.93**

Mean and capped-sum alternatives were rejected: mean dilutes strong sources, capped-sum has an arbitrary ceiling.

### 3. `non_apprehension` ≠ `not_enough_evidence`

Two superficially similar concepts must stay distinct at every schema level:

| Property | `non_apprehension` (evidence type) | `not_enough_evidence` (verdict) |
|---|---|---|
| Meaning | Sensor-confirmed absence of an entity/state | Insufficient textual sources to verify claim |
| Source | AI2THOR only (closed-world) | AVeriTeC (open-world) |
| `evidence[].stance` | `"absent"` | `"not_enough_evidence"` |
| Weight | 0.75 — closed-world absence is a positive epistemic act | 0.75 via EW lookup |

`non_apprehension` is never assigned to AVeriTeC records — web text cannot confirm closed-world absence. `evidence[].stance = "absent"` is AI2THOR-only. Merging these would corrupt stance labels and misrepresent the epistemic situation.

---

## Consequences

- Confidence priors are theoretically grounded and citable in the research paper
- Every adapter must map its evidence to these categories — a non-trivial but consistent requirement
- Weights are heuristic priors, not empirically validated; model experiments will test whether they improve performance vs. uniform weights
- The combination formula is active in `EC_i = 1 - (1-ST_i)^(EW_i × IS_i)` at graph build time
