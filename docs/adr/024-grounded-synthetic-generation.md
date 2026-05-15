# ADR-024: Grounded Synthetic Generation — Seed Pool and AI2THOR Triplets

## Status

Accepted

## Context

The `LocalTextClient` (ADR-023) generates fictional text via random vocabulary substitution. While sufficient for pipeline validation, it produces repetitive text whose SBERT embeddings cluster artificially. This risks the GNN learning surface-level embedding patterns from the synthetic data rather than the epistemic features.

Two improvements were identified:

1. **Semantic grounding**: use realistic (claim, evidence) pairs rather than random entity substitution
2. **Structured triplets**: AI2THOR is already an RDF knowledge graph — its claims carry `(subject, predicate, object)` triples that can populate the `evidence.triples` field in synthetic records

Additionally, the seed pool file location and the output filename convention for synthetic batches needed an architectural decision.

## Decision

### Seed pool (`data/registry/seed_pool.jsonl`)

A hand-curated JSONL file of ~25 fictional (claim, supporting_evidence, refuting_evidence) pairs, covering all five evidence types and multiple domains (household appliances, civic policy, research findings, environmental statistics, etc.).

**Location rationale:** `data/registry/` not `data/raw/synthetic/`. The seed pool is reference data, not pipeline output:
- It is hand-authored and version-controlled
- It should not be included in the `build` step's synthetic glob
- It belongs alongside `source_trust_registry.jsonl` — both are curated reference inputs

Schema per record:
```json
{
  "claim": "...",
  "evidence_type": "testimony|perception|inference|comparison_analogy|non_apprehension",
  "domain": "...",
  "supporting_evidence": "...",
  "refuting_evidence": "..."
}
```

The `GroundedClient` samples from this pool by `evidence_type`, then applies reliability perturbations:
- `strong` stance = use evidence text as-is
- `weak` stance = prepend a hedging phrase ("Reportedly, ", "Allegedly, " etc.)
- Multiple supporting items = add connectors ("Additionally, ", "Corroborating this, ")

### AI2THOR triplet integration

For templates using `perception` or `non_apprehension` evidence types, `GroundedClient` loads `data/raw/ai2thor/claims_all.jsonl` and pools records by stance (`supports`, `refutes`, `absent`).

When generating for a perception/non_apprehension spec, the client:
1. Samples an AI2THOR record matching the required stance
2. Uses the AI2THOR claim text and evidence text
3. Returns `evidence_triples` from the AI2THOR record alongside the text

This gives perception/non_apprehension synthetic records real `(subject, predicate, object)` triples (e.g., `["Lettuce|...", "isDirty", "False"]`) rather than `triples: []`. The `triple_source` field is set to `"ai2thor_simulation"`.

For non-perception types (testimony, inference, comparison_analogy), the text seed pool is used and `triples` remains `[]` — consistent with AVeriTeC records.

### Fixed output filename

`generate-synthetic` always writes to `data/raw/synthetic/synthetic_current.jsonl`, overwriting the previous batch. The `build` step checks for this single file rather than globbing `*.jsonl` (which would inadvertently include seed_pool.jsonl if it were in the same directory).

Version history is managed by git, not by accumulating timestamped files.

## Consequences

**Positive:**
- Claim–evidence pairs are semantically coherent — evidence discusses the claim
- Perception/non_apprehension records carry real AI2THOR triples, exercising the GNN's triplet processing path
- Seed pool in `data/registry/` is clearly distinguished from generated output
- Fixed output filename makes `build` deterministic and removes the glob bug
- Fallback chain: GroundedClient → LocalTextClient when pool has no matching record

**Negative:**
- Seed pool has ~25 records — sampling with replacement creates duplicate claims at scale; acceptable for current 1,000-record batches but will require expansion for larger datasets
- AI2THOR claims are simulator-grounded (object properties, spatial states) — linguistic register differs from AVeriTeC text claims; SBERT embedding space may separate by register rather than epistemic content
- Fixed output means old batches are overwritten without warning; users must use git to recover previous versions
