# ADR-014: Graph Node and Edge Schema

## Status

Accepted

## Context

Each unified JSONL record must be converted to a PyG `HeteroData` object for GNN training. The schema must:

1. Represent the two structurally different source types (AI2THOR structured triples; AVeriTeC text evidence) within the same graph type
2. Expose the Pramana confidence weight as a relational property (not a node property)
3. Preserve the semantic meaning of evidence stance (`supports`, `refutes`, `absent`, `no_evidence`) — not collapse it to a scalar
4. Include raw modality information per evidence item so Phase 5 can test whether the model can learn Pramana types from modality patterns
5. Carry per-claim metadata needed for Phase 6 per-Pramana and per-source evaluation

## Decision

### Node types (4)

| Node type | Count per record | Feature vector | Notes |
|---|---|---|---|
| `claim` | 1 | sentence-transformer embedding (384-d) | One per record; verdict label attached here |
| `evidence` | 1 per evidence item | embedding (384-d) + modality one-hot (5-d) = **389-d** | Pathway B adds inferred Pramana one-hot (5-d) → 394-d |
| `epistemic` | 1 | pramana_primary one-hot (5-d) + confidence_weight (1-d) = **6-d** | Bridges source-specific encoders and shared classifier (ADR-004) |
| `triple` | 1 per claim triple | embedding of `"s p o"` string (384-d) | **AI2THOR only** — absent for AVeriTeC (`claim_triples=null`) |

Modality one-hot encoding (5 values): `simulation_state=0`, `web_text=1`, `video=2`, `audio=3`, `image=4`.
Pramana one-hot encoding (5 values, in TRAINING_PRAMANA order): `perception=0`, `testimony=1`, `non_apprehension=2`, `comparison_analogy=3`, `inference=4`.

### Edge types (8)

| Edge type | Direction | Edge attr | Condition |
|---|---|---|---|
| `(claim, has_evidence, evidence)` | claim → evidence | 1.0 | all records |
| `(evidence, supports, claim)` | evidence → claim | 1.0 | `evidence.stance == "supports"` |
| `(evidence, refutes, claim)` | evidence → claim | 1.0 | `evidence.stance == "refutes"` |
| `(evidence, absent, claim)` | evidence → claim | 1.0 | `evidence.stance == "absent"` — AI2THOR only: simulation confirmed physical absence |
| `(evidence, no_evidence, claim)` | evidence → claim | 1.0 | `evidence.stance == null` (JSON null / Python `None`) — AVeriTeC NEE: web search found no answer |
| `(claim, has_epistemic, epistemic)` | claim → epistemic | `confidence_weight` | all records |
| `(claim, has_triple, triple)` | claim → triple | 1.0 | AI2THOR only |
| `(evidence, from_triple, triple)` | evidence → triple | 1.0 | AI2THOR only |

### Graph-level metadata (attached to `HeteroData`)

| Field | Type | Purpose |
|---|---|---|
| `data.y` | int tensor | Verdict label: 0=supported, 1=refuted, 2=not_enough_evidence |
| `data.pramana` | str | `epistemic.pramana_primary` — used in Phase 6 per-Pramana evaluation |
| `data.dataset` | str | `provenance.dataset` (`ai2thor` or `averitec`) — used in Phase 6 per-source evaluation |

## Consequences

**Why stance must be 4 separate edge types (not a scalar weight):**
A single `(evidence, ?, claim)` edge with stance encoded as a scalar (e.g., supports=1.0, absent=0.5, refutes=0.0) collapses directional semantics into a magnitude signal. `HeteroConv` learns a distinct GATConv weight matrix per edge type — using separate edge types lets the model learn that refuting evidence propagates a contradictory signal, which cannot be expressed as a weaker version of supporting evidence.

**Why `confidence_weight` is `edge_attr` on `has_epistemic`, not a node feature:**
The confidence weight is a property of the *relationship* between a claim and its Pramana type, not an intrinsic property of either the claim node or the epistemic node. Encoding it as `edge_attr` lets GATConv scale the epistemic message at the edge level, which is the correct inductive bias for ADR-004's architecture.

**Why `evidence.modality` is on EvidenceNode, not only on EpistemicNode:**
The ADR-007 heuristic rules map per-evidence modalities to the claim-level `pramana_primary`. But each evidence item carries its own modality — collapsing this to the claim level loses resolution. Including modality one-hot on EvidenceNode gives the model raw evidence-level signal. In Pathway B (Phase 5), the model can use this per-evidence modality to learn Pramana inference, testing whether learned patterns match the ADR-007 heuristic.

**Why TripleNodes only for AI2THOR:**
All AVeriTeC records have `claim_triples=null` — there are no structured triples. The `"s p o"` text embedding approach is sufficient for Phase 4. A structured mini triple-graph (entity nodes + predicate edges) within each AI2THOR subgraph is a Phase 5 ablation.

**Why `absent` and `no_evidence` are distinct edge types (not merged):**
`absent` is AI2THOR confirming physical absence — the simulation state was checked and the object was not found. Verdict is `supported` because a claim like "There is no vase in this scene" is positively confirmed. `no_evidence` is AVeriTeC NEE — a web search query returned no usable answer. Verdict is `not_enough_evidence` because the claim cannot be verified either way. Merging them into a single edge type would conflate a positive confirmation of absence with an unresolvable search result, destroying the epistemic distinction between `non_apprehension` (absence-confirmed, ADR-005) and `not_enough_evidence` (search-unresolved).

In the unified schema, `absent` arrives as the string `"absent"` in `evidence[i].stance`; `no_evidence` arrives as JSON `null` (Python `None`). The graph builder preserves `None` directly so `s is None` correctly routes to `no_evidence` edges — the `or ""` coercion that would mask `None` values must not be used when reading stances.

**Why `pramana_all` multi-Pramana EpistemicNodes are deferred:**
The schema stores `pramana_all` for multi-Pramana claims. One EpistemicNode per entry in `pramana_all` (each with its individual prior weight) would let the model reason over multiple epistemic pathways simultaneously. This is a Phase 5 ablation — Phase 4 uses one EpistemicNode per claim keyed to `pramana_primary` for simplicity and debuggability.

**Stance-verdict determinism and Phase 5 requirement:**
The four stance edge types encode the verdict with 100% rule accuracy on the Phase 4 training set:

```
any no_evidence edge  →  not_enough_evidence  (verdict=2)
any refutes edge      →  refuted              (verdict=1)
else                  →  supported            (verdict=0)
```

This means the Phase 4 Pathway A baseline learns edge-type routing rather than epistemic reasoning — val_acc reaches 1.0 from epoch 1. The GNN does not need to read claim text, embeddings, or the epistemic node; the edge types alone determine the verdict.

The epistemic hypothesis ("Pramana-aware structure improves fact-verification") cannot be tested with stance edges present. Phase 5 must include a stance-removal ablation: the model predicts verdict from claim text and epistemic node only, with evidence nodes present but all stance back-edges removed. Any accuracy above the random baseline (33.3% for 3-class) under this condition is evidence that epistemic structure carries independent signal. The recommended Phase 5 ablation matrix:

| Run | Stance edges | Epistemic node | Tests |
|---|---|---|---|
| A | removed | absent | random-like floor |
| B | removed | present | Pramana contribution |
| C | present (Pathway A) | present | stance-routing ceiling |
| D | present (Pathway B) | learned from modality | learned vs. heuristic Pramana |
