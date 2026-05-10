# ADR-002: Unified Null-Tolerant Schema v2.0

## Status

Accepted

## Context

The two primary data sources have fundamentally different evidence structures:

- **AI2-THOR** produces structured triples (`claim_triples`, `evidence[].triples`), typed reasoning (`one_hop`, `conjunction`), and graph-native data
- **AVeriTeC** produces unstructured text evidence (QA pairs, web text), no triples, no reasoning structure

Two design approaches were considered:

1. **Separate schemas per dataset** — one strict schema for AI2-THOR, one for AVeriTeC; the GNN model would need dataset-specific input heads
2. **Single unified schema with null-tolerant fields** — one schema where AI2-THOR-specific fields (`claim_triples`, `reasoning`) are allowed to be `null` for AVeriTeC

## Decision

Use a **single null-tolerant schema** (v2.0, defined at `data/schema/unified_schema.json`).

AI2-THOR-only fields are explicitly nullable in the JSON Schema definition. The fields are always present as keys (never absent), but their values are `null` for AVeriTeC records. This is enforced at conversion time by the adapters and validated against the schema.

Fields dropped entirely (rather than nulled) during the v2.0 redesign:
- `claim_meta` — source-specific metadata with no GNN signal
- `qa` structure — AVeriTeC QA pairs are flattened into `evidence[].text`
- `verdict.annotator_confidence` — always null in both sources
- `epistemic.proof_type_rationale` — always null in both sources
- `evidence[].cached_source_url` — engineering artifact with no training value

## Consequences

**Positive:**
- One validator, one downstream schema consumer, one GNN input pipeline
- Adding a new dataset requires only implementing the two ABCs — no schema changes needed if the new source fits the existing field set
- Simpler tooling: `validate_unified_dataset.py` handles all sources in one pass

**Negative:**
- ~50% of records have `null` for `claim_triples` and `reasoning` (all AVeriTeC records) — GNN must handle sparse inputs
- The schema appears to support more fields than any single record actually uses, which can mislead contributors about what is guaranteed to be non-null
- Future datasets with truly different structures (e.g., video evidence) may require schema evolution or extension

**Mitigation:**
The `provenance.dataset` field in every record tells downstream consumers exactly which source the record came from, so field nullness can be predicted deterministically rather than checked per-record.
