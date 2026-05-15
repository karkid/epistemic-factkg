# ADR-011: Per-Evidence Labeling Rules

**Status:** Accepted
**Merges:** ADR-011 (AI2THOR rules + AVeriTeC rules)
**Related:** ADR-001 (Pramana weights), ADR-009 (source trust registry), ADR-010 (per-evidence modeling)

---

## Context

Schema v3.0 assigns `evidence_types`, `source_id`, and `inference_strength` per evidence item rather than per claim (ADR-010). Both AI2THOR and AVeriTeC adapters need deterministic rules to produce these fields from their respective raw formats.

---

## AI2THOR Rules

AI2THOR exposes two structural dimensions per claim:
- `reasoning.strategy` — the verification mechanism per evidence item
- `reasoning.structural` — the logical shape of the claim (one_hop, conjunction, etc.)

**Use `strategy`, not `structural`, to assign `evidence_types`.** Strategy encodes the epistemic mechanism (how evidence was gathered). Structural encodes claim shape (what is asserted). A conjunction of two direct observations is still `[perception, perception]` — no inference step is involved.

### Strategy → evidence_types

| Strategy | `evidence_types` | Rationale |
|---|---|---|
| `direct_observation` | `["perception"]` | Sensor directly reads the property |
| `absence_detection` | `["perception", "non_apprehension"]` | Scan (perception) + confirmed absence |
| `spatial_reasoning` | `["perception", "comparison_analogy"]` | Sensor reads positions; verifying spatial relation requires comparison |
| `action_testing` | `["perception", "inference"]` | Affordance tested by attempting action; outcome is inferred |
| fallback (no strategy, has triples) | `["perception"]` | |
| fallback (no strategy, no triples) | `["non_apprehension"]` | |

### Fixed fields for all AI2THOR evidence items

| Field | Value |
|---|---|
| `source_id` | `"ai2thor_simulation"` |
| `inference_strength` | `1.0` — simulator state is directly read; zero inference steps |
| `modality` | `"simulation_state"` |

### Absence claims: `claim_triples = null`

Absence claims assert non-existence — there is no positive triple. `claim_triples = null` is correct and expected; validators must not flag it as missing.

---

## AVeriTeC Rules

AVeriTeC provides per-answer `source_medium`, `answer_type`, and claim-level `fact_checking_strategies`. Assignment uses four passes per evidence item.

### Pass 1 — Modality base types

| Modality | `evidence_types` |
|---|---|
| `image`, `video`, `audio` | `["perception"]` |
| `web_table` | `["comparison_analogy", "testimony"]` |
| `web_text`, `pdf`, `annotator_knowledge`, `other` | `["testimony"]` |
| `unanswerable` | `[]` |

### Pass 2 — Numeric cue detection (textual items only)

If the answer text contains numeric/statistical cues (`%`, `GDP`, `million`, `rank`, `largest`, etc.) → append `comparison_analogy`.

### Pass 3 — Multi-source abstractive inference

If **distinct source URLs across the claim ≥ 2** AND `answer_type == abstractive` → append `inference`. A single-source abstractive answer is a summary, not multi-hop inference.

### Pass 4 — `fact_checking_strategies` enrichment (textual items only)

| Strategy | Appends |
|---|---|
| `Numerical Comparison` | `comparison_analogy` |
| `Consultation` | `inference` |

Perceptual items are excluded: strategy labels describe what the annotator did, not the nature of the observation. `Written Evidence` is not mapped — testimony is already assigned by Pass 1.

### `inference_strength` heuristics

| `answer_type` | IS |
|---|---|
| `boolean`, `extractive` | 0.8 — direct one-step lookup |
| `abstractive` | 0.6 — synthesis, potentially multi-step |
| `unanswerable` | 0.0 |

### `source_id` resolution

Resolved from `source_url` via the source trust registry (ADR-009):
1. Parse domain; strip `www.`
2. Exact registry match: `{name}_{modality}`
3. Subdomain strip: try parent domain
4. TLD heuristic: `.gov` → `government_web_text`, `.edu` → `general_web_text`
5. Social media check
6. Modality default (`web_table`, `pdf`)
7. Fallback: `unknown_web` (ST = 0.40)

---

## Shared Consequences

- `non_apprehension` is never assigned to AVeriTeC — web text cannot confirm closed-world absence (ADR-001)
- `evidence_types_all` at the claim level is the union across all items — used for dataset filtering and GNN readiness checks
- AI2THOR labeling is fully deterministic; AVeriTeC labeling is heuristic but rule-based
- Reasoning strategy (`reasoning.strategy`) is the correct granularity for AI2THOR — it is set per-evidence item, so individual items in a conjunction claim can have different strategies
