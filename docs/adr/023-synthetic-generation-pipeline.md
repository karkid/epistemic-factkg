# ADR-023: Synthetic Generation Pipeline Design

## Status

Accepted

## Context

ADR-022 established the need for shortcut-breaking synthetic records. The generation pipeline must:

1. Guarantee correct EC-formula verdicts by construction (not by LLM judgment)
2. Cover all five training evidence types: testimony, perception, inference, comparison_analogy, non_apprehension
3. Generate compound/multi-evidence records with asymmetric trust levels
4. Work offline (no API key required) and produce linguistically varied text
5. Be extensible — swap text clients without changing the epistemic layer

## Decision

**Two-layer architecture:** separate the *epistemic* layer (template configs + EC formula) from the *linguistic* layer (text generation clients).

### Template configs

Each template is a `_TemplateConfig` with a list of `EvidenceSpec` entries. Each spec fixes:
- `stance` — contributes to support_score or refute_score
- `source_id` — resolves to ST via the source trust registry
- `evidence_types` — determines EW via `combine_pramana_weights`
- `inference_strength` — IS in the EC formula
- `reliability` — hint to the text client: `"strong"` | `"weak"` | `"hedged"` | `"absent"`

The verdict is derived deterministically from the template's EC math. The LLM or local client provides only the natural-language text.

### 15 templates (as of v3.0)

| Template | Evidence items | Verdict | Shortcut-breaking? |
|---|---|---|---|
| high_trust_supported | 2 testimony IS=0.8 | supported | no |
| low_trust_nee | 1 testimony IS=0.6, social_media | NEE | ✓ supports→NEE |
| high_trust_refuted | 2 testimony IS=0.8 | refuted | no |
| low_trust_refuted_nee | 1 testimony IS=0.5, unknown_web | NEE | ✓ refutes→NEE |
| conflicting | 1 supports + 1 refutes IS=0.8 | conflicting | ✓ |
| strong_support_weak_refute | 2S IS=0.8 + 1R IS=0.4 social | supported | ✓ has R stance |
| weak_support_strong_refute | 1S IS=0.6 social + 2R IS=0.8 | refuted | ✓ has S stance |
| weak_vs_weak_nee | 1S IS=0.5 + 1R IS=0.4, both low-trust | NEE | ✓ both stances |
| corroborating_3 | 3 testimony IS=0.7-0.8 | supported | no |
| perception_direct | 1 perception IS=1.0 ai2thor | supported | no |
| inference_nee | 2 inference IS=0.5 academic | NEE | ✓ supports→NEE |
| comparison_supported | 2 comparison_analogy IS=0.7 | supported | no |
| non_apprehension_absent | 1 non_apprehension IS=0.8 ai2thor | supported | no |
| non_apprehension_refuted | 1 non_apprehension IS=0.8 ai2thor, refutes | refuted | no |
| non_apprehension_weak_nee | 1 non_apprehension IS=0.6 general_web | NEE | ✓ absent→NEE |

Total shortcut-breaking fraction: ~62%.

### Pluggable text clients

```
SyntheticTextClient (abstract)
├── LocalTextClient     — vocabulary pools + random substitution, fully offline
├── GroundedClient      — seed pool (data/registry/seed_pool.jsonl) + AI2THOR triplets
└── LLMClient           — Anthropic API (claude-haiku; highest linguistic diversity)
```

All clients implement `generate(specs, template_name) → {"claim": str, "evidence_texts": [...], "evidence_triples": [...]}`.

**Default client selection (CLI):**
1. If `ANTHROPIC_API_KEY` set → `LLMClient`
2. If seed pool exists → `GroundedClient`
3. Else → `LocalTextClient`

### `FictionalClaimGenerator`

The orchestrator: owns templates, calls the client, runs `_build_record` which applies the EC formula and assembles the v3.0 record. `support_score` and `refute_score` are NOT stored — they are computed at model-build time from the stored evidence items and registry.

### Validation

`SyntheticDataValidator.validate_batch()` checks:
- Shortcut fraction ≥ 35%
- All evidence items have v3.0 required fields (`evidence_types`, `source_id`, `inference_strength`)
- EC values are recomputable from stored fields (spot-check consistency)

## Consequences

**Positive:**
- EC math is guaranteed correct — the template drives the verdict, not the LLM
- All 5 evidence types covered by dedicated templates
- Compound asymmetric cases (`strong_support_weak_refute` etc.) explicitly force aggregation learning
- Three interchangeable clients allow offline validation, grounded text, and full LLM-quality generation
- `support_score`/`refute_score` removed from stored records — single source of truth at model-build time

**Negative:**
- 15 templates require maintenance when the EC formula changes (threshold values are baked in)
- LocalTextClient text is repetitive — SBERT embeddings may cluster artificially; only suitable for pipeline validation, not the final training dataset
- Template distribution tuning is manual — no automatic optimisation for GNN learning signal
