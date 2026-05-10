# ADR-005: Anupalabdhi (non_apprehension) Is Distinct from not_enough_evidence

## Status

Accepted

## Context

Two superficially similar concepts appear in the data:

1. **`non_apprehension`** — the Pramana of confirmed absence. In AI2-THOR, the simulator explicitly confirms that an object or state is *not present* in the scene. This is a positive epistemic act — the absence is observed, not inferred.
2. **`not_enough_evidence`** — an AVeriTeC verdict label. It means textual sources were insufficient to support or refute a claim. The absence here is epistemic limitation, not observed absence.

There was a temptation to collapse both into a single category (e.g., a generic "absence" bucket) since both concern the lack of confirming evidence.

## Decision

Treat them as **distinct at every level of the schema**:

| Property | `non_apprehension` | `not_enough_evidence` |
|---|---|---|
| Type | Pramana label (epistemic category) | Verdict label (claim outcome) |
| Meaning | Sensor-confirmed ABSENCE | Insufficient textual sources |
| Source | AI2-THOR only | AVeriTeC only |
| `evidence[].stance` | `"absent"` | `"supports"` or omitted |
| `epistemic.pramana_primary` | `"non_apprehension"` | `"non_apprehension"` (triggered by verdict) |
| Confidence | 0.75 (closed-world confirmed) | 0.75 (triggered prior) |

When an AVeriTeC record has verdict `not_enough_evidence`, the `infer_pramana()` logic assigns `pramana_primary = "non_apprehension"` — because the epistemic situation (absence of confirming knowledge) maps to the same Pramana category. But `evidence[].stance` is NOT set to `"absent"` for AVeriTeC records, since the absence is not sensor-confirmed.

## Consequences

**Positive:**
- The GNN can distinguish between "confirmed absent" (AI2-THOR) and "unverifiable" (AVeriTeC) at the evidence level, even though both carry the same Pramana prior
- Prevents a subtle bug where `evidence[].stance = "absent"` would be wrongly assigned to AVeriTeC claims, corrupting the graph edge type
- Maintains a clean separation between the Pramana layer (epistemology) and the verdict layer (claim outcome)

**Negative:**
- Two things share the same `pramana_primary` value for different reasons — requires understanding the distinction to avoid confusion
- The mapping (`not_enough_evidence` verdict → `non_apprehension` pramana) is not obvious and must be documented here and in [ADR-007](007-heuristic-epistemic-labeling.md)
