# ADR-013: Synthetic Data Pipeline

**Status:** Accepted
**Merges:** ADR-013 (pipeline design + grounded generation and seed pool)
**Related:** ADR-012 (shortcut leakage rationale), ADR-009 (source trust registry)

---

## Context

ADR-012 established the need for shortcut-breaking synthetic records. The pipeline must:
1. Guarantee correct EC-formula verdicts by construction (not by LLM judgment)
2. Cover all five training evidence types
3. Work offline (no API key required)
4. Be extensible — swap text clients without changing the epistemic layer

---

## Decision

### Two-layer architecture

Separate the **epistemic layer** (template configs + EC formula verdict derivation) from the **linguistic layer** (text generation clients).

Each template is a `_TemplateConfig` with a list of `EvidenceSpec` entries. Each spec fixes `stance`, `source_id`, `evidence_types`, `inference_strength`, and a `reliability` hint for the text client. The verdict is derived deterministically from the template's EC math. The client provides only natural-language text.

### 15 templates

| Template | Evidence | Verdict | Shortcut-breaking? |
|---|---|---|---|
| `high_trust_supported` | 2 testimony IS=0.8 | supported | no |
| `low_trust_nee` | 1 testimony IS=0.6 social_media | NEE | ✓ supports→NEE |
| `high_trust_refuted` | 2 testimony IS=0.8 | refuted | no |
| `low_trust_refuted_nee` | 1 testimony IS=0.5 unknown_web | NEE | ✓ refutes→NEE |
| `conflicting` | 1 supports + 1 refutes IS=0.8 | conflicting | ✓ |
| `strong_support_weak_refute` | 2S IS=0.8 + 1R IS=0.4 social | supported | ✓ has R stance |
| `weak_support_strong_refute` | 1S IS=0.6 social + 2R IS=0.8 | refuted | ✓ has S stance |
| `weak_vs_weak_nee` | 1S IS=0.5 + 1R IS=0.4, both low-trust | NEE | ✓ both stances |
| `corroborating_3` | 3 testimony IS=0.7–0.8 | supported | no |
| `perception_direct` | 1 perception IS=1.0 ai2thor | supported | no |
| `inference_nee` | 2 inference IS=0.5 academic | NEE | ✓ supports→NEE |
| `comparison_supported` | 2 comparison_analogy IS=0.7 | supported | no |
| `non_apprehension_absent` | 1 non_apprehension IS=0.8 ai2thor | supported | no |
| `non_apprehension_refuted` | 1 non_apprehension IS=0.8 ai2thor, refutes | refuted | no |
| `non_apprehension_weak_nee` | 1 non_apprehension IS=0.6 general_web | NEE | ✓ absent→NEE |

Shortcut-breaking fraction: ~62%. Minimum required: 35% (enforced by `SyntheticDataValidator`).

### Pluggable text clients

```
SyntheticTextClient (abstract)
├── LocalTextClient    — vocabulary pools + random substitution; fully offline
├── GroundedClient     — seed pool + AI2THOR triplets (default when seed pool exists)
└── LLMClient          — Anthropic API (claude-haiku); highest linguistic diversity
```

All clients implement `generate(specs, template_name) → {"claim": str, "evidence_texts": [...], "evidence_triples": [...]}`.

CLI default selection: `ANTHROPIC_API_KEY` set → `LLMClient`; seed pool exists → `GroundedClient`; else → `LocalTextClient`.

### Grounded client — seed pool

A hand-curated JSONL at `data/registry/seed_pool.jsonl` (~25 fictional claim/evidence pairs covering all five evidence types and multiple domains).

**Location: `data/registry/`, not `data/raw/synthetic/`.** The seed pool is reference data, not pipeline output — it is hand-authored, version-controlled, and must not be included in the build step's synthetic input glob.

Schema per record: `{ "claim", "evidence_type", "domain", "supporting_evidence", "refuting_evidence" }`.

`GroundedClient` samples by `evidence_type`, then applies reliability perturbations:
- `strong` → use text as-is
- `weak` → prepend hedging phrase ("Reportedly, ", "Allegedly, ")
- Multiple supporting items → add connectors ("Additionally, ", "Corroborating this, ")

### Grounded client — AI2THOR triplet integration

For templates using `perception` or `non_apprehension`, `GroundedClient` loads `data/raw/ai2thor/claims_all.jsonl` and pools records by stance. When generating for a perception/non_apprehension spec, it samples a matching AI2THOR record and returns its `(subject, predicate, object)` triples as `evidence_triples`. The `triple_source` field is set to `"ai2thor_simulation"`.

For other evidence types, `triples` remains `[]` — consistent with AVeriTeC records.

### Fixed output filename

`just synthetic` always writes to `data/raw/synthetic/synthetic_current.jsonl`, overwriting the previous batch. Version history is managed by git. This makes `just build` deterministic and removes any glob-over-seed-pool bug.

### Validation

`SyntheticDataValidator.validate_batch()` checks:
- Shortcut fraction ≥ 35%
- All evidence items have v3.0 required fields
- EC values are recomputable from stored fields (spot-check)

---

## Consequences

- EC math is guaranteed correct — the template drives the verdict, not the LLM
- `support_score`/`refute_score` are NOT stored in records — computed at model-build time from evidence items + registry; single source of truth
- `LocalTextClient` text is repetitive; use only for pipeline validation, not the final training dataset
- Seed pool has ~25 records — sampling with replacement creates duplicates at scale; acceptable for current 1,000-record batches
- Fixed output filename means old batches are overwritten; use git to recover previous versions
