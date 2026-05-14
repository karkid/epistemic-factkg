# ADR-020: AI2THOR Evidence-Type Labeling Rules

**Status:** Accepted  
**Date:** 2026-05-15  
**Supersedes:** Portions of ADR-007 (AI2THOR-specific rules)  
**Related:** ADR-019 (per-evidence epistemic modeling), ADR-005 (non_apprehension)

---

## Context

AI2THOR records expose two structural dimensions per claim:
- **`reasoning.structural`** — the *shape* of the claim (one_hop, conjunction, negation, absence)
- **`evidence[].strategy`** — the *mechanism* used to verify it (direct_observation, absence_detection, spatial_reasoning, action_testing)

An earlier approach (ADR-007) assigned `pramana_primary` at the claim level based on structural type. Schema v3.0 moves epistemic labels to per-evidence items (`evidence[].evidence_types`), requiring a decision about which dimension drives the assignment.

---

## Decision

**Use `strategy` (per-evidence), not `structural` (per-claim), to assign `evidence_types`.**

### Rationale

`strategy` directly encodes the epistemic mechanism — *how* the evidence was gathered. `structural` encodes the claim's logical shape — *what* the claim asserts. The two are orthogonal:

- A `conjunction` claim (two facts joined by AND) may have two `direct_observation` evidence items, each perception only. Conjunction is not multi-hop; no inference step is involved.
- A `negation` claim (negated property) is still verified by direct sensor observation → `direct_observation` strategy → `[perception]`.
- `absence_detection` represents sensor scanning + failure to find → two epistemic acts, hence two types.

Assigning from structural would conflate claim shape with epistemic quality, making the labels inaccurate.

### Strategy → evidence_types mapping

| Strategy | `evidence_types` | Rationale |
|---|---|---|
| `direct_observation` | `["perception"]` | Sensor directly reads the property |
| `absence_detection` | `["perception", "non_apprehension"]` | Scan (perception) + confirmed absence (non_apprehension) |
| `spatial_reasoning` | `["perception", "comparison_analogy"]` | Sensor reads positions; verifying spatial relation requires comparison |
| `action_testing` | `["perception", "inference"]` | Affordance tested by attempting the action; outcome is inferred from the attempt |

**Fallback** (no strategy field): `has_ev_triples → ["perception"]`; empty triples → `["non_apprehension"]`.

### Fixed fields for all AI2THOR evidence items

| Field | Value | Reason |
|---|---|---|
| `source_id` | `"ai2thor_simulation"` | Closed-world ground-truth source |
| `inference_strength` | `1.0` | Simulator state is directly read; zero inference steps |
| `modality` | `"simulation_state"` | Not web text or sensory media |

### Absence claims: why `claim_triples = null`

Absence claims assert non-existence. There is no positive triple to write (the asserted entity is absent). `claim_triples = null` is correct and expected for these records; validators must not flag it as missing.

Refuted absence claims (the object *was* found) similarly have `claim_triples = null` — the corruption removes the claim triple to represent the refuted scenario.

---

## Consequences

- Evidence types for AI2THOR are fully deterministic and automated; no human annotation required.
- `conjunction` claims produce two evidence items with `["perception"]` each — correctly reflecting two independent direct observations.
- `absence_detection` producing `["perception", "non_apprehension"]` models the two-step epistemic process: scan then confirm absence. This matches ADR-005's requirement that non_apprehension requires a closed-world state.
- Strategy is the correct granularity: it is set per-evidence item in the v2.0 ClaimGenerator output, so individual items in a conjunction claim can in principle have different strategies.
