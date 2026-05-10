# Data Flow

This document traces how data moves through the pipeline — from raw source inputs to final split JSONL files ready for GNN training.

## Pipeline Overview

```
AI2-THOR simulator                   AVeriTeC
       │                                  │
    [build]                         (manual download)
  ① build_rdf                            │
       │                                  │
 out/knowledge_graph.ttl      data/raw/averitec/{train,dev}.json
       │
  ② build_claims
       │
 data/raw/ai2thor/claims_all.jsonl
       │                                  │
       └──────── ③ convert_to_unified ────┘
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

**Commands:** `just build` runs steps ①②③ in sequence. `just run` runs build → validate → report with timestamped logs.

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

Queries the RDF graph to produce supported and refuted claims. Claims are typed by reasoning structure (`one_hop`, `conjunction`, `negation`, `absence`) and annotated with the source triples. Each record is in AI2-THOR's raw internal format at this stage — not yet in unified schema.

---

## Step 3: Convert to Unified Schema v2.0

**Command:** `just build` (third sub-step — runs `src.cli.convert_to_unified`)

**Input:**
- `data/raw/ai2thor/claims_all.jsonl`
- `data/raw/averitec/train.json`
- `data/raw/averitec/dev.json`

**Output:** `out/unified/epistemic_factkg.jsonl`

Each source adapter (`AI2ThorConverter`, `AveritecConverter`) applies `infer_pramana()` to assign Pramana labels and confidence weights, then calls `convert_one()` to produce a unified v2.0 record. The final JSONL concatenates all sources.

Key conversions:
- AI2-THOR: `claim_triples`, `reasoning`, `evidence[].triple_source = "ground_truth"` are populated; `pramana_primary` is `perception` or `non_apprehension`
- AVeriTeC: `claim_triples = null`, `reasoning = null`, QA pairs flattened into `evidence[].text`; `pramana_primary` is `testimony` or `inference`

See [ADR-002](adr/002-unified-schema-v2-null-tolerant.md) for schema design and [ADR-007](adr/007-heuristic-epistemic-labeling.md) for labeling strategy.

### Unified Record Schema (key fields)

```json
{
  "schema_version": "2.0",
  "id": "...",
  "claim": "...",
  "verdict": { "label": "supported|refuted|not_enough_evidence|conflicting_evidence", "justification": "..." },
  "epistemic": {
    "pramana_primary": "perception|testimony|inference|...",
    "pramana_all": ["..."],
    "confidence_weight": 0.95,
    "assignment_method": "rule_based"
  },
  "claim_triples": [["subject", "predicate", "object"]] ,
  "reasoning": { "structural": "one_hop|conjunction|...", "strategy": "..." },
  "evidence": [{
    "evidence_id": "...",
    "text": "...",
    "triples": null,
    "triple_source": "ground_truth|extracted|null",
    "modality": "simulation_state|web_text|pdf|...",
    "stance": "supports|refutes|absent",
    "source_url": null
  }],
  "provenance": { "dataset": "ai2thor|averitec", "split": null, "context_id": "..." },
  "meta": { "schema_version": "2.0", "created_utc": "..." }
}
```

Full schema definition: `data/schema/unified_schema.json`

---

## Step 4: Validate

**Command:** `just validate`

**Input:** `out/unified/epistemic_factkg.jsonl`

**Output:** `out/report/validation.json`

Runs three layers of validation:
1. **Schema validation** — JSON Schema Draft-07 against `data/schema/unified_schema.json`
2. **Dataset-level semantic checks** — adapter-specific validators (`AI2ThorValidator`, `AveritecValidator`) check field consistency (e.g., AI2-THOR claims must have non-null triples)
3. **Pramana checks** — confidence weights must match known Pramana values; `pramana_primary` must be in `pramana_all`

---

## Step 5: Split (AI2-THOR only)

**Command:** `uv run python -m src.cli.split_ai2thor --input <file> --output_dir <dir>`

**Input:** AI2-THOR subset of unified JSONL (or the full file)

**Output:**
- `<output_dir>/ai2thor_train.jsonl`
- `<output_dir>/ai2thor_dev.jsonl`
- `<output_dir>/ai2thor_test.jsonl`
- `<output_dir>/ai2thor_splits_manifest.json`

Splits by `provenance.context_id` (floorplan), not by claim index. Default: 80/10/10 by floorplan count (`--mode pct`). All claims from a given floorplan go to exactly one split.

See [ADR-009](adr/009-floorplan-based-train-test-split.md) for why floorplan-based split was chosen over random split.

AVeriTeC already comes pre-split (train/dev/test) from the original dataset — no re-splitting needed.

---

## Step 6: Dataset Report

**Command:** `just report`

**Input:** `out/report/validation.json`

**Output:** `out/report/` (markdown report + charts)

Generates a summary report with per-dataset statistics, Pramana distribution, verdict distribution, and validation error breakdown.
