# ADR-010: Per-Evidence Epistemic Modeling

## Status

Accepted (partially superseded by ADR-028: absent stance removed, aggregation updated)

## Context

Schema v2.0 assigned a single `pramana_primary` label at the **claim** level. This created a
shortcut: because all evidence items for a claim shared the same Pramana type, the GNN could
learn to predict the verdict from the claim-level Pramana alone, ignoring evidence confidence
entirely. The symptom was near-perfect stance→verdict correlation: `evidence.stance = supports`
implied `verdict = supported` in ≈100% of training records.

Two related problems:
1. **Epistemic type is a property of individual evidence items, not the claim.** A claim can be
   supported by both direct observation (perception) and indirect inference; these have different
   epistemic weights.
2. **Source trust was implicit and uniform.** A tweet and a Reuters article contributing to the
   same claim received identical weight, making confidence weights uninformative.

This ADR records the formula and labeling rules introduced to fix both problems.

## Decision

### Move epistemic labels to evidence level

`pramana_primary` is abolished. Each evidence item now carries:

| Field | Type | Description |
|---|---|---|
| `evidence_types` | `string[]` | Multi-label EvidenceType classification for this item |
| `source_id` | `string` | Registry key → $ST_i$ via source trust registry (ADR-009) |
| `inference_strength` | `float` | $IS_i$ — directness of evidence-to-claim inference (0.0–1.0) |

`evidence_types_all` at the claim level is a summary union (for dataset filtering), not a GNN
input.

### Per-evidence confidence formula (Weighted Product)

$$EC_i = 1 - (1 - ST_i)^{EW_i \times IS_i}$$

This is the **Weighted Product** formula: the exponent $EW_i \times IS_i$ scales the complement
$(1 - ST_i)$, so that higher epistemic weight or stronger inference raises confidence
multiplicatively rather than additively. Equivalently, it asks: *given $EW_i \times IS_i$
independent draws from a source with reliability $ST_i$, what is the probability of at least one
success?* This formulation was chosen over a simple product ($ST_i \times EW_i \times IS_i$)
because the multiplicative interaction of the three factors better models how epistemic quality
and source credibility jointly limit confidence — weak source trust cannot be fully compensated by
high epistemic type weight alone.

- $ST_i$ — source trustworthiness, resolved from registry via `source_id`
- $EW_i$ — epistemic-type weight: `combine_evidence_weights(evidence_types)` — diminishing-returns
  combination over the item's evidence types (ADR-001)
- $IS_i$ — inference strength rubric (see below)

**EC_i floor:** If $EC_i < 0.10$, the evidence contributes negligibly — converters override stance
to `not_enough_evidence`.

### Inference strength ($IS_i$) rubric

| IS | Description |
|---|---|
| 1.0 | Direct ground truth (simulator state, primary measurement) |
| 0.8 | One-step extractive (direct quote, Boolean answer, official record) |
| 0.6 | Abstractive / multi-source synthesis |
| 0.4 | Partial or circumstantial (covers only part of claim) |
| 0.2 | Speculative or highly hedged |
| 0.0 | Unanswerable |

IS is a property of **evidence content**, not source quality. A highly-trusted source with an
abstractive answer still gets IS = 0.6; $ST_i$ captures source quality separately.

### Aggregation

$$SupportScore = 1 - \prod_{i \in \text{supports}}(1 - EC_i)$$
$$RefuteScore = 1 - \prod_{i \in \text{refutes}}(1 - EC_i)$$

Evidence with `not_enough_evidence` or `conflicting_evidence` stance is excluded from both
aggregations. Absence (non_apprehension) claims carry `supports` stance directly (ADR-028) —
no special case needed in the aggregation.

### Verdict derivation thresholds

```
supported            : support_score >= 0.75 AND refute_score < 0.40
refuted              : refute_score  >= 0.75 AND support_score < 0.40
conflicting_evidence : support_score >= 0.40 AND refute_score >= 0.40
not_enough_evidence  : everything else
```

**Threshold rationale:**
- 0.75 = high-confidence belief threshold: requires at least one strong piece of evidence (EC ≈
  0.75) or several moderate ones combined
- 0.40 = non-trivial opposition threshold: a refute_score below this means opposition is too weak
  to block "supported" (and vice versa)
- The strict `<` on the 0.40 check means a refute score exactly at 0.40 blocks "supported" —
  erring on the side of caution when opposition is non-trivial

### EvidenceType and EvidenceStance naming

`Pramana` is renamed to `EvidenceType` (same six values). A backward-compatible alias
`Pramana = EvidenceType` is retained until converters are updated in subsequent steps.

`EvidenceStance` gains two new values:
- `not_enough_evidence` — evidence exists but is ambiguous, hedged, or incomplete
- `conflicting_evidence` — a single evidence item internally contradicts itself

These were previously represented as `null` stances in v2.0; making them explicit allows
converters to emit them deterministically.

### Implementation

All formula functions, constants, and registry utilities live in
`src/core/claims/labels.py`:

| Symbol | Purpose |
|---|---|
| `compute_evidence_confidence(st, ew, is_)` | EC_i formula |
| `combine_evidence_weights(types)` | EW_i (diminishing returns, ADR-001) |
| `aggregate_scores(evidence_items, registry)` | (support_score, refute_score) |
| `derive_verdict(support_score, refute_score)` | Verdict label |
| `SUPPORT_THRESHOLD = 0.75` | |
| `REFUTE_THRESHOLD = 0.75` | |
| `CONFLICT_FLOOR = 0.40` | |
| `MIN_EVIDENCE_CONFIDENCE = 0.10` | EC_i floor for stance override |

## Consequences

**Positive:**
- Shortcut is broken: same stance + low source trust → low EC_i → verdict may be
  `not_enough_evidence` even when all stances are `supports`
- EC_i is interpretable: GNN edge weights carry genuine epistemic signal
- `EvidenceStance` is now fully enumerated; no null stances in valid records
- Formula is fully deterministic given `source_id`, `evidence_types`, `inference_strength`

**Negative:**
- Converters must assign `source_id` and `inference_strength` per evidence item, not per claim
- IS rubric is heuristic; abstractive vs extractive distinction requires NLP or manual annotation
- Verdict thresholds (0.75, 0.40) are chosen analytically; empirical calibration against
  annotated gold labels is deferred to post-training evaluation

**Future work:**
- Calibrate thresholds against AVeriTeC dev-set verdict gold labels after initial training
- Replace IS heuristics with a learned inference-directness classifier
