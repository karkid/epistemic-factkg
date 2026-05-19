# Data Flow

This document traces how data moves through the pipeline — from raw source inputs to final split JSONL files ready for GNN training.

## Pipeline Overview

```
AI2-THOR simulator          AVeriTeC              Seed pool + AI2THOR claims
       │                       │                          │
    [build]              (manual download)        [generate-synthetic]
  ① build_rdf                  │                          │
       │                       │                          │
 out/knowledge_graph.ttl  data/raw/averitec/        data/raw/synthetic/
       │                 {train,dev}.json          synthetic_current.jsonl
  ② build_claims                │                          │
       │                        │                          │
 data/raw/ai2thor/              │                          │
 claims_all.jsonl               │                          │
       │                        │                          │
       └──────────── ③ convert_to_unified ─────────────────┘
                              │
               out/unified/epistemic_factkg.jsonl
                              │
                        [validate]
                              │
               out/report/validation.json
                              │
                         [report]
                              │
                    out/report/ (md + charts)
```

**Commands:** `just build` runs ①②③ in sequence (requires `synthetic_current.jsonl` to exist). `just rebuild` calls `just generate-synthetic` first, then `just build`. `just run` runs build → validate → report with timestamped logs.

**Source split (v3.0 target):** ~28% AI2THOR (~1,800) / ~56% AVeriTeC (3,568) / ~16% Synthetic (~1,000). See [ADR-012](adr/012-dataset-composition-and-generation-strategy.md).

---

## Step 0: Generate Synthetic Records (shortcut-breaking)

**Command:** `just generate-synthetic`

**Inputs:**
- `data/registry/seed_pool.jsonl` — 25 hand-curated (claim, evidence) seed pairs by evidence type
- `data/raw/ai2thor/claims_all.jsonl` — used as real triplet source for perception/non_apprehension templates
- `configs/config.yaml` → `synthetic.n_records`, `synthetic.distribution`
- `ANTHROPIC_API_KEY` (optional) — if set, uses `LLMClient`; else uses `GroundedClient` (seed pool + AI2THOR) or `LocalTextClient`

**Output:** `data/raw/synthetic/synthetic_current.jsonl` (overwrites on each run; version via git)

Generates fictional shortcut-breaking claims using 15 templates. Each template fixes the EC-formula math deterministically — the text client provides only the linguistic layer. ~62% of records break the stance→verdict shortcut (same stance, different verdict depending on source trust and inference strength).

Key templates:

| Category | Templates | Shortcut-breaking? |
|---|---|---|
| High-trust clear-verdict | `high_trust_supported`, `high_trust_refuted` | No |
| Low-trust → NEE | `low_trust_nee`, `low_trust_refuted_nee` | Yes |
| Asymmetric compound | `strong_support_weak_refute`, `weak_support_strong_refute`, `weak_vs_weak_nee` | Yes |
| Conflicting | `conflicting` | Yes |
| Perception (AI2THOR) | `perception_direct` | No |
| Inference → NEE | `inference_nee` | Yes |
| Non-apprehension | `non_apprehension_absent`, `non_apprehension_refuted`, `non_apprehension_weak_nee` | Partial |

See [ADR-022](adr/022-shortcut-leakage-and-synthetic-data-strategy.md), [ADR-023](adr/023-synthetic-generation-pipeline.md), [ADR-024](adr/024-grounded-synthetic-generation.md).

---

## Step 1: Build Knowledge Graph

**Command:** `just build` (first sub-step — runs `src.cli.build_rdf`)

**Input:** `configs/config.yaml`

**Output:** `out/knowledge_graph.ttl`

Reads the AI2-THOR config (scene types, object randomization) and runs the simulation to produce an RDF/Turtle knowledge graph of all objects, properties, and spatial relations across configured floorplans. Each triple has the form `<entity_uri> <predicate> <value>`.

See [ADR-010](adr/010-rdf-as-kg-intermediate-format.md) for why RDF/Turtle was chosen.

---

## Step 2: Generate AI2-THOR Claims

**Command:** `just build` (second sub-step — runs `src.cli.build_claims`)

**Input:** `out/knowledge_graph.ttl`

**Output:** `data/raw/ai2thor/claims_all.jsonl`

Queries the RDF graph to produce supported and refuted claims. Claims are typed by reasoning structure (`one_hop`, `conjunction`, `negation`, `absence`) and annotated with the source triples. Each record is output directly in unified schema v3.0 — `modality: "sensor"`, `assignment_method: "simulator"`, `evidence_types` from strategy, no intermediate format.

---

## Step 3: Convert to Unified Schema v3.0

**Command:** `just build` (third sub-step — runs `src.cli.convert_to_unified`)

**Inputs:**
- `data/raw/ai2thor/claims_all.jsonl`
- `data/raw/averitec/train.json`
- `data/raw/averitec/dev.json`
- `data/raw/synthetic/synthetic_current.jsonl`
- `data/registry/source_trust_registry.jsonl`

**Output:** `out/unified/epistemic_factkg.jsonl`

Each source adapter applies per-evidence epistemic labeling — setting `evidence_types`, `source_id`, `inference_strength` — then calls `convert_one()` to produce a unified v3.0 record. Evidence types are per-evidence and multi-label; `pramana_primary` no longer exists.

Key per-adapter behavior:
- **AI2THOR**: Generator outputs v3.0 directly. `evidence[].evidence_types` is set from `reasoning.strategy`: `direct_observation` → `["perception"]`; `absence_detection` → `["perception", "non_apprehension"]`; `spatial_reasoning` → `["perception", "comparison_analogy"]`; `action_testing` → `["perception", "inference"]`. `source_id = "sensor_perception"`, `inference_strength = 1.0`, `modality = "sensor"`, `assignment_method = "simulator"`. `claim_triples` and `evidence[].triples` populated from RDF.
- **AVeriTeC**: `evidence_types` from modality + answer-type heuristics; `source_id` resolved from `source_url` domain; `inference_strength` from answer type (extractive=0.8, abstractive=0.6); `claim_triples = null`
- **Synthetic**: evidence fields already set by template; `triple_source = "ai2thor_simulation"` for perception/non_apprehension records with real triples; `provenance.dataset = "synthetic"`

The **EC formula** is applied in all converters:

$$EC_i = 1 - (1 - ST_i)^{EW_i \times IS_i}$$

where $ST_i$ is looked up from `source_trust_registry.jsonl`, $EW_i = \text{combine\_evidence\_weights(evidence\_types)}$, and $IS_i$ is `inference_strength`.

Verdict aggregation uses product-of-complements over supporting and refuting evidence items:

$$\text{SupportScore} = 1 - \prod_{i \in \text{Supports}}(1 - EC_i)$$
$$\text{RefuteScore} = 1 - \prod_{i \in \text{Refutes}}(1 - EC_i)$$

For synthetic records, the verdict is set by template construction (not re-derived); `derivation_method = "aggregated_from_evidence"`. For AI2THOR and AVeriTeC, the original annotated verdict is kept; `derivation_method = "annotated"`.

See [ADR-019](adr/019-per-evidence-epistemic-modeling.md) for per-evidence epistemic modeling and [ADR-015](adr/015-source-trust-registry.md) for the source trust registry design.

### Unified Record Schema (key fields, v3.0)

```json
{
  "schema_version": "3.0",
  "id": "...",
  "claim": "...",
  "verdict": {
    "label": "supported|refuted|not_enough_evidence|conflicting_evidence",
    "justification": null,
    "derivation_method": "aggregated_from_evidence|annotated"
  },
  "epistemic": {
    "evidence_types_all": ["testimony", "inference"],
    "assignment_method": "simulator|heuristic|annotated|llm_generated"
  },
  "claim_triples": [["subject", "predicate", "object"]],
  "reasoning": { "structural": "one_hop|conjunction|...", "strategy": "..." },
  "evidence": [{
    "evidence_id": "...",
    "text": "...",
    "triples": null,
    "triple_source": "ground_truth|ai2thor_simulation|extracted|null",
    "modality": "sensor|web_text|pdf|...",
    "stance": "supports|refutes|not_enough_evidence|conflicting_evidence",
    "source_url": null,
    "evidence_types": ["testimony"],
    "source_id": "bbc_web_text",
    "inference_strength": 0.8
  }],
  "provenance": { "dataset": "ai2thor|averitec|synthetic", "split": null, "context_id": "..." },
  "meta": { "schema_version": "3.0", "created_utc": "..." }
}
```

Full schema definition: `src/epistemic/schema.py` (`CLAIM_SCHEMA`)

---

## Step 4: Validate

**Command:** `just validate`

**Input:** `out/unified/epistemic_factkg.jsonl`

**Output:** `out/report/validation.json`

Runs three layers of validation:
1. **Schema validation** — JSON Schema Draft-07 against `CLAIM_SCHEMA` in `src/epistemic/schema.py`
2. **Dataset-level semantic checks** — adapter-specific validators check field consistency (e.g., AI2-THOR claims must have non-null triples; all evidence items must have `evidence_types`, `source_id`, `inference_strength`)
3. **Epistemic checks** — shortcut fraction ≥ 35% in synthetic records; `evidence_types_all` matches union of per-evidence types; stance values are within the valid enum

---

## Step 5: Split

**Command:** `just split`

**Input:** `out/training/epistemic_factkg_training.jsonl`

**Output:**
- `out/splits/train_indices.json`
- `out/splits/val_indices.json`
- `out/splits/test_indices.json`

Default: 80/10/10 by record index with stratified verdict labels (seed=42). AI2THOR records are split by `provenance.context_id` (floorplan) to avoid cross-split data leakage from the same scene. AVeriTeC and synthetic records are split by stratified random sampling.

See [ADR-009](adr/009-floorplan-based-train-test-split.md) for the split strategy design.

---

## Step 6: Dataset Report

**Command:** `just report`

**Input:** `out/report/validation.json`

**Output:** `out/report/` (markdown report + charts)

Generates a summary report with per-dataset statistics, evidence type distribution, verdict distribution (per source), shortcut fraction audit, and validation error breakdown.

---

## Reference Data

Two curated reference files live in `data/registry/` — they are inputs to the pipeline, not pipeline outputs:

| File | Purpose |
|---|---|
| `data/registry/source_trust_registry.jsonl` | Maps `source_id` → `source_trust` (ST) for the EC formula; ~90 entries covering AI2THOR, common AVeriTeC domains, and synthetic source types |
| `data/registry/seed_pool.jsonl` | 25 hand-authored (claim, evidence) pairs used by `GroundedClient` to produce semantically coherent synthetic text |
