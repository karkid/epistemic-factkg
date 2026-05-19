# ADR-028: Generator Direct v3.0 Output — Schema Corrections and Stance Redesign

**Status:** Accepted
**Supersedes (partially):** ADR-001 (absent stance), ADR-010 (aggregation), ADR-011 (AI2THOR labeling rules)
**Related:** ADR-009 (source trust registry), ADR-010 (per-evidence modeling)

---

## Context

Several accumulated issues in the pipeline required a coordinated fix:

1. **Two-step generation**: `ClaimGenerator` emitted v2.0 JSONL (with `pramana_primary`,
   `pramana_all`, no `evidence_types`, wrong modality). A separate `AI2ThorConverter` step
   transformed it to v3.0. This meant mapping logic was split across two places, and the raw
   JSONL on disk was never schema-valid.

2. **`absent` stance was a leaky abstraction**: `EvidenceStance.ABSENT` was treated as a
   synonym for "sensor-confirmed absence supports the claim". In practice this was logically
   redundant — an absence claim that is supported is supported; the stance is `supports`.
   Keeping a separate `absent` value forced every downstream consumer (formula, model,
   validator, synthetic data) to explicitly handle an extra case.

3. **`simulation_state` modality was ambiguous**: The term described where the data came from
   (a simulation), not the channel through which the evidence was perceived. `sensor` is the
   correct epistemic descriptor — the simulator behaves as a sensor.

4. **`assignment_method: "rule_based"` was too generic**: AI2THOR evidence is assigned by the
   simulator itself, not by a post-hoc rule. `"simulator"` is the correct value.

5. **Schema allowed null for required fields**: `verdict.label` and `evidence.text` accepted
   null, which meant invalid records could pass validation silently.

---

## Decisions

### 1. Generator outputs v3.0 directly

The `ClaimGenerator` / `ClaimInstance.get_schema_layout()` now outputs schema v3.0 records.
Mapping logic (strategy → evidence_types, stance, modality, source_id) is embedded during
generation, not as a post-processing step.

A new shared module `src/adapters/ai2thor/claims/strategy.py` holds the mapping helpers
(`_classify_strategy`, `_infer_evidence_types`, `_label_to_stance`, etc.). Both the generator
and converter import from there — no logic duplication.

`AI2ThorConverter` is now a lightweight normaliser for legacy data. For new v3.0 records it
only decodes URIs and normalises label strings.

### 2. `EvidenceStance.ABSENT` removed

`EvidenceStance.ABSENT` is removed from the enum. Valid stances are now:
`supports`, `refutes`, `not_enough_evidence`, `conflicting_evidence`.

**Absence (non_apprehension) claims** follow the same stance rule as all other claims:
- Verdict `supported` (object confirmed absent) → stance `supports`
- Verdict `refuted` (object IS present, contradicting claim) → stance `refutes`

This is logically correct: stance answers "does this evidence support the claim?" — a confirmed
absence supports a claim of absence, so the stance is `supports`. There is no need for a
separate symbol.

**Aggregation impact**: `formula.aggregate_scores()` no longer special-cases `absent`.
Absence evidence with `supports` stance flows naturally into `SupportScore`.

**Model impact**: `STANCE_TO_INT` drops `"absent": 0`. No functional change — it was mapped
to the same int as `"supports"`.

### 3. Modality `simulation_state` → `sensor`

All AI2THOR evidence uses `modality: "sensor"`. Updated in:
- `schema.py` enum
- `types.py` generator output
- `converter.py` legacy path
- `source_trust_registry.jsonl` AI2THOR entry
- `model/data/types.py` `MODALITY_TO_INT`

`sensor` is the epistemic descriptor: the AI2THOR simulator acts as a sensor that reads scene
state. The prior term `simulation_state` described implementation, not epistemics.

### 4. `assignment_method: "simulator"` for AI2THOR

All AI2THOR records use `epistemic.assignment_method = "simulator"`. The schema enum gains
`"simulator"` as a valid value alongside `heuristic`, `rule_based`, `annotated`, `llm_generated`.

### 5. `verdict.label` and `evidence.text` are non-null required strings

Both fields are now `"type": "string"` (not `["string", "null"]`) in `schema.py`. Every claim
must have a verdict label; every evidence item must have a text description. For
triples-only AI2THOR evidence, `text` falls back to the realized claim string.

### 6. Absence claims always carry `evidence_types: ["perception", "non_apprehension"]`

The fallback in the old ADR-011 table assigned `["non_apprehension"]` alone to absence
claims. This has been corrected: the sensor must *perceive* the absence — perception is always
present. The correct types are `["perception", "non_apprehension"]` for all
`absence_detection` strategy claims.

The old fallback `(no strategy, no triples) → ["non_apprehension"]` is also updated to
`["perception", "non_apprehension"]` for consistency.

---

## Updated AI2THOR Labeling Rules (supersedes ADR-011 AI2THOR section)

| Strategy | `evidence_types` | Notes |
|---|---|---|
| `direct_observation` | `["perception"]` | Sensor reads property directly |
| `absence_detection` | `["perception", "non_apprehension"]` | Scan (perception) + confirmed absence |
| `spatial_reasoning` | `["perception", "comparison_analogy"]` | Position reading + spatial comparison |
| `action_testing` | `["perception", "inference"]` | Affordance tested; outcome inferred |
| fallback (no strategy, has triples) | `["perception"]` | |
| fallback (no strategy, no triples) | `["perception", "non_apprehension"]` | Absence implied |

**Fixed fields for all AI2THOR evidence items:**

| Field | Value |
|---|---|
| `source_id` | `"sensor_perception"` |
| `inference_strength` | `1.0` |
| `modality` | `"sensor"` |
| `assignment_method` | `"simulator"` |
| `stance` | `"supports"` or `"refutes"` (mirrors verdict) |

---

## Files Changed

| File | Change |
|---|---|
| `src/epistemic/enums.py` | Removed `EvidenceStance.ABSENT` |
| `src/epistemic/schema.py` | `sensor` modality, `simulator` assignment_method, non-null label + text |
| `src/epistemic/formula.py` | Removed `ABSENT` from aggregation stance check |
| `src/epistemic/validator.py` | Updated absence stance validation rules |
| `src/adapters/ai2thor/claims/strategy.py` | New — shared mapping helpers |
| `src/adapters/ai2thor/claims/types.py` | `Evidence.reasoning_strategy` field; `get_schema_layout()` → v3.0 |
| `src/adapters/ai2thor/claims/generator.py` | Passes `reasoning_strategy`; absence sites use `"absence_detection"` |
| `src/adapters/ai2thor/converter.py` | Imports from `strategy.py`; simplified `_from_v3` pass-through |
| `src/adapters/ai2thor/validator.py` | Removed `ABSENT` rules; expanded valid evidence types |
| `src/model/data/types.py` | `"sensor"` in `MODALITY_TO_INT`; removed `"absent"` from `STANCE_TO_INT` |
| `src/adapters/synthetic/fictional_generator.py` | `absent` stance → `supports`/`not_enough_evidence` |
| `data/registry/source_trust_registry.jsonl` | AI2THOR entry `modality` → `sensor` |

---

## Consequences

- Raw AI2THOR JSONL on disk is now schema v3.0-valid without a converter pass
- `AI2ThorConverter` is still required for legacy records and URI normalisation
- Existing `data/raw/ai2thor/claims_all.jsonl` must be regenerated (`just build rebuild=true`)
- `EvidenceStance.ABSENT` is fully removed; any code that referenced it will raise `AttributeError` immediately rather than silently using a stale value
- The aggregation formula is simpler: one fewer stance case
