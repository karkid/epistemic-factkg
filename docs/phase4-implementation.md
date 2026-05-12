# Phase 4: Graph Construction & Model Development

**Status:** Complete  
**Date:** 2026-05-12  
**ADRs:** 013, 014, 015

---

## Overview

Phase 4 converts the epistemic-annotated JSONL records from Phase 3 into a heterogeneous
graph dataset and trains a GNN that reasons over epistemic structure — not just text.

The core research question this phase sets up: *does making the model aware of the epistemic
type of a claim (perception vs. testimony vs. non-apprehension) improve fact-verification accuracy?*

---

## 1. Architecture Decisions

### ADR-015 — Verdict Class Reduction (3-class)

The original dataset had 4 verdict classes. `conflicting_evidence` (233 records, 4.3%) was
dropped for training. It is an AVeriTeC artifact with no clean Pramana mapping and too few
records to learn reliably.

| Class | Label | Count | Weight |
|---|---|---|---|
| `supported` | 0 | ~2,003 (39.0%) | 0.91 |
| `refuted` | 1 | ~3,153 (61.4%) | 0.58 |
| `not_enough_evidence` | 2 | ~339 (6.6%) | **5.40** |

Class imbalance is handled with **inverse-frequency weighted CrossEntropyLoss** — no rebuild
needed. The NEE class is heavily upweighted (5.4×) because it is central to the
`non_apprehension` Pramana contribution (ADR-005).

**Random baseline (3-class): 33.3%. Target: >50% after 10 epochs.**

---

## 2. Graph Schema (ADR-014)

Each claim becomes one heterogeneous subgraph — there are no cross-claim edges.

### 2.1 Node Types

```
┌─────────────────────────────────────────────────────────────────┐
│                       Node Feature Vectors                       │
├──────────────┬───────────┬─────────────────────────────────────┤
│  Node Type   │    Dim    │  Composition                         │
├──────────────┼───────────┼─────────────────────────────────────┤
│  claim       │   384-d   │  sentence embedding                  │
│  evidence    │   389-d   │  sentence embedding (384)            │
│              │           │  + modality one-hot (5)              │
│  epistemic   │     6-d   │  pramana one-hot (5)                 │
│              │           │  + confidence_weight (1)             │
│  triple      │   384-d   │  "s p o" string → embedding          │
│              │           │  [AI2THOR only; 0-row for AVeriTeC]  │
└──────────────┴───────────┴─────────────────────────────────────┘
```

**Sentence embeddings** use `all-MiniLM-L6-v2` (384-d) via `sentence-transformers`.
Embeddings are cached to `out/graphs/embed_cache.pkl` keyed by SHA-256 hash — subsequent
runs skip re-embedding.

**Modality one-hot** (5-d) on each EvidenceNode:

| Index | Modality | Source |
|---|---|---|
| 0 | `simulation_state` | AI2THOR |
| 1 | `web_text` | AVeriTeC |
| 2 | `video` | (reserved) |
| 3 | `audio` | (reserved) |
| 4 | `image` | (reserved) |

**Pramana one-hot** (5-d) on EpistemicNode:

| Index | Pramana | Meaning |
|---|---|---|
| 0 | `perception` | Direct sensory evidence |
| 1 | `testimony` | Web-sourced text |
| 2 | `non_apprehension` | Absence evidence |
| 3 | `comparison_analogy` | Analogical reasoning |
| 4 | `inference` | Derived logical conclusion |

### 2.2 Edge Types

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Edge Schema                                  │
├──────────────────────────────────┬──────────────┬───────────────────┤
│  Edge Type                       │  edge_attr   │  Notes            │
├──────────────────────────────────┼──────────────┼───────────────────┤
│  (claim, has_evidence, evidence) │  —           │  all records      │
│  (evidence, supports, claim)     │  —           │  stance=supports  │
│  (evidence, refutes, claim)      │  —           │  stance=refutes   │
│  (evidence, absent, claim)       │  —           │  stance=absent    │
│                                  │              │  AI2THOR only:    │
│                                  │              │  absence confirmed│
│  (evidence, no_evidence, claim)  │  —           │  stance=null      │
│                                  │              │  AVeriTeC NEE:    │
│                                  │              │  search no answer │
│  (claim, has_epistemic,          │  confidence_ │  all records;     │
│          epistemic)              │  weight [1]  │  prior from ADR-8 │
│  (claim, has_triple, triple)     │  —           │  AI2THOR only     │
│  (evidence, from_triple, triple) │  —           │  AI2THOR only     │
└──────────────────────────────────┴──────────────┴───────────────────┘
```

**Why stance = 4 separate edge types (not a scalar weight):**  
`HeteroConv` learns a different `GATConv` weight matrix per edge type. Separate
`supports` / `refutes` / `absent` / `no_evidence` types let the model learn that
refuting evidence *contradicts* the claim, absence evidence *confirms* a negation,
and missing evidence *leaves the verdict unresolved* — a single scalar-weighted edge
collapses all of this to magnitude only.

**Why `absent` and `no_evidence` are not merged:**  
`absent` = AI2THOR simulation confirmed the object is physically absent → verdict is
`supported` (a "there is no X" claim is confirmed). `no_evidence` = AVeriTeC web search
returned no usable answer → verdict is `not_enough_evidence`. Same Pramana type
(`non_apprehension`), opposite verdicts — merging them would destroy this distinction.

**Why `confidence_weight` as `edge_attr`:**  
The Pramana prior (ADR-008) is a property of the *relationship* between claim and
epistemic type, not of either node alone. Passing it as `edge_attr` lets GATConv
scale the message at the edge level.

**Schema consistency guarantee:**  
Every `HeteroData` object always contains all 8 edge types. Edge types with no edges
in a given graph (e.g. `has_triple` for AVeriTeC records) are initialized with an empty
`edge_index` tensor `[2, 0]`. Without this, PyG's `InMemoryDataset` creates shorter-than-N+1
slices for sparse edge types, breaking `__getitem__` at the first gap graph.

---

## 3. Graph Diagram (per claim)

```
                    ┌─────────────────────────────────────────┐
                    │           CLAIM NODE  (384-d)           │
                    │      "The apple is on the table."       │
                    └────────┬───────────┬────────────────────┘
                             │           │
              has_epistemic  │           │  has_evidence       has_triple
              (w=0.95)       │           │  ──────────────────────────────
                             ▼           ▼                               │
              ┌──────────────────┐   ┌──────────────────┐               │
              │  EPISTEMIC NODE  │   │  EVIDENCE NODE   │               │
              │      (6-d)       │   │     (389-d)       │               │
              │ pramana: percep. │   │ "The apple is on  │               │
              │ weight: 0.95     │   │  the table."      │               │
              └──────────────────┘   │ modality: sim [0] │               │
                                     └────────┬──────────┘               │
                                              │ supports                  │
                                              │ (stance=supports)         │
                                              ▼                           ▼
                                         claim node             ┌──────────────────┐
                                         (back edge)            │   TRIPLE NODE    │
                                                                │     (384-d)      │
                                                                │ "entity:Apple    │
                                                                │  isOn            │
                                                                │  entity:Table"   │
                                                                └──────────────────┘
                                                                [AI2THOR only]
```

**AVeriTeC example** (no triples, refutes stance):

```
     ┌────────────────────────────────────────┐
     │          CLAIM NODE  (384-d)            │
     │   "The president signed the bill."     │
     └──────────┬───────────┬────────────────┘
                │            │
  has_epistemic │            │ has_evidence
   (w=0.80)     │            │
                ▼            ▼
    ┌──────────────┐   ┌──────────────────────┐
    │  EPISTEMIC   │   │   EVIDENCE NODE 0    │
    │  testimony   │   │  "The president      │
    │  w=0.80      │   │   vetoed the bill."  │
    └──────────────┘   │   modality: web[1]   │
                       └────────┬─────────────┘
                                │ refutes
                                ▼
                           claim node
                           (back edge)

                       ┌──────────────────────┐
                       │   EVIDENCE NODE 1    │
                       │  "No signing         │
                       │   ceremony held."    │
                       │   modality: web[1]   │
                       └────────┬─────────────┘
                                │ refutes
                                ▼
                           claim node
                           (back edge)
```

---

## 4. GNN Architecture (ADR-013)

### 4.1 EpistemicHGNN

```
Input Node Features (variable dim per type)
         │
         │  claim:     384-d
         │  evidence:  389-d   (384 + 5 modality)
         │  epistemic:   6-d   (5 pramana + 1 weight)
         │  triple:    384-d
         ▼
┌─────────────────────────────────────────────┐
│      Linear Projections (per node type)     │
│    claim/evidence/epistemic/triple → 256-d  │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│   HeteroConv Layer 1  (GATConv per edge)    │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  (claim, has_evidence, evidence)    │   │
│  │  GATConv(256→256, heads=2,          │   │
│  │          concat=False)              │   │
│  ├─────────────────────────────────────┤   │
│  │  (evidence, supports, claim)        │   │
│  │  GATConv(256→256, heads=2)          │   │
│  ├─────────────────────────────────────┤   │
│  │  (evidence, refutes, claim)         │   │
│  │  GATConv(256→256, heads=2)  ← learns│   │
│  │           contradiction signal      │   │
│  ├─────────────────────────────────────┤   │
│  │  (evidence, absent, claim)          │   │
│  │  GATConv(256→256, heads=2)          │   │
│  ├─────────────────────────────────────┤   │
│  │  (evidence, no_evidence, claim)     │   │
│  │  GATConv(256→256, heads=2)          │   │
│  ├─────────────────────────────────────┤   │
│  │  (claim, has_epistemic, epistemic)  │   │
│  │  GATConv(256→256, heads=2,          │   │
│  │          edge_dim=1)   ← conf.weight│   │
│  ├─────────────────────────────────────┤   │
│  │  (claim, has_triple, triple)        │   │
│  │  GATConv(256→256, heads=2)          │   │
│  ├─────────────────────────────────────┤   │
│  │  (evidence, from_triple, triple)    │   │
│  │  GATConv(256→256, heads=2)          │   │
│  └─────────────────────────────────────┘   │
│        aggregation: mean                   │
└─────────────────────────────────────────────┘
         │
         ▼  ReLU + Dropout(0.3)
         │
┌─────────────────────────────────────────────┐
│   HeteroConv Layer 2  (same structure)      │
│   2nd hop: claim → evidence → triple paths  │
└─────────────────────────────────────────────┘
         │
         ▼  ReLU + Dropout(0.3)
         │
         ├──────────────────────────────────────────────
         │  claim node embedding [B, 256]
         ▼
┌──────────────────────────────────────────────┐
│  Verdict Classifier                          │
│  Linear(256→128) → ReLU → Dropout(0.3)       │
│  → Linear(128→3)                             │
│  → logits [B, 3]                             │
│  (supported=0, refuted=1, NEE=2)             │
└──────────────────────────────────────────────┘

         [Pathway B only — Phase 5 ablation]
         ├──────────────────────────────────────────────
         │  epistemic node embedding [B, 256]
         ▼
┌──────────────────────────────────────────────┐
│  Pramana Aux Head                            │
│  Linear(256→5)                               │
│  → logits [B, 5]  (Pramana type prediction)  │
└──────────────────────────────────────────────┘
```

### 4.2 Architecture Rationale

| Choice | Reason |
|---|---|
| **HeteroConv** over homogeneous GNN | 4 structurally different node types, 8 semantically different edge types — a single representation loses epistemic structure |
| **GATConv** over SAGEConv | SAGEConv has no `edge_attr` support (can't pass `confidence_weight`) and no attention weights (Phase 6 explainability not possible) |
| **GATConv** over HGT | HGT adds type-specific projection matrices — too many parameters for ~5k records; high overfitting risk |
| **2 layers** | Layer 1 = claim↔evidence; Layer 2 = claim→evidence→triple. More layers cause over-smoothing on small subgraphs (~6 nodes avg) |
| **heads=2, concat=False** | Dataset too small for high head count; `concat=False` keeps output at 256-d (average across heads) |
| **Per-claim subgraphs** | No cross-claim edges exist; one big graph would create false structural links between unrelated claims |

### 4.3 Parameter Count (approximate)

| Component | Parameters |
|---|---|
| Projections (4 types × ~256 output) | ~265k |
| HeteroConv layer 1 (8 GATConvs × ~131k each) | ~1,048k |
| HeteroConv layer 2 | ~1,048k |
| Verdict classifier (256→128→3) | ~33k |
| **Total (Pathway A)** | **~2.4M** |
| + Pramana head (Pathway B) | +1.3k |

---

## 5. Two Epistemic Pathways

Phase 4 builds infrastructure for both pathways. Pathway B runs as a Phase 5 ablation.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Pathway A — Prior-Given (Phase 4 baseline)                          │
│                                                                       │
│  EpistemicNode.x = [pramana_one_hot | confidence_weight]   (6-d)    │
│  Pramana label is a fixed heuristic from ADR-007.                    │
│  confidence_weight is a fixed prior from ADR-008.                    │
│                                                                       │
│  → Model uses pre-computed epistemic knowledge as input facts.       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Pathway B — Modality-Learned (Phase 5 ablation)                     │
│                                                                       │
│  EvidenceNode.x includes modality one-hot (dims 384-388).            │
│  Auxiliary Pramana head on EpistemicNode predicts Pramana type       │
│  from the attended evidence modalities.                              │
│                                                                       │
│  Enabled with: --use-modality-learning  (train_gnn.py)               │
│  Loss: verdict_loss + aux_loss_weight × pramana_loss                 │
│                                                                       │
│  → Model learns to deduce epistemic type from raw evidence signal.   │
└──────────────────────────────────────────────────────────────────────┘

Phase 6 question: does the model learn the same patterns as the heuristic
rules (ADR-007), or discover different (better) ones?
```

---

## 6. Data Pipeline

```
out/unified/epistemic_factkg.jsonl        (5,368 records — Phase 3)
            │
            │  just filter
            │  src/cli/filter_for_training.py
            │  excludes: postulation_derivation (ADR-011)
            │            conflicting_evidence   (ADR-015)
            ▼
out/training/epistemic_factkg_training.jsonl   (5,135 records)
            │
            │  just build-graph
            │  src/cli/build_graph_dataset.py
            │  → ClaimGraphBuilder (graph_builder.py)
            │  → Featurizer (all-MiniLM-L6-v2, cache to embed_cache.pkl)
            │  → EpistemicFactDataset (InMemoryDataset)
            ▼
out/graphs/graph_dataset.pt                    (5,135 HeteroData graphs)
out/graphs/embed_cache.pkl                     (SHA-256 → embedding cache)
            │
            │  just split
            │  src/cli/split_dataset.py
            │  AI2THOR:  floorplan split by context_id (ADR-009)
            │  AVeriTeC: stratified random split by verdict (seed=42)
            ▼
out/splits/train_indices.json   (4,106 = 80.0%)
out/splits/val_indices.json     (  512 = 10.0%)
out/splits/test_indices.json    (  517 = 10.1%)
            │
            │  just train
            │  src/cli/train_gnn.py
            │  → EpistemicHGNN (model.py)
            │  → Trainer (train.py)
            │    weighted CrossEntropyLoss
            │    Adam(lr=1e-3)
            │    ReduceLROnPlateau(patience=5)
            │    early stopping (patience=10)
            ▼
out/checkpoints/best_model.pt
out/checkpoints/training_history.json
```

---

## 7. Dataset Statistics

| Metric | Value |
|---|---|
| Total graphs | 5,135 |
| Training graphs | 4,106 (80.0%) |
| Validation graphs | 512 (10.0%) |
| Test graphs | 517 (10.1%) |
| Split seed | 42 (deterministic) |
| Node types per graph | 4 (always) |
| Edge types per graph | 8 (always, empty tensors for missing) |
| Sentence embedding dim | 384 |
| Embedding cache entries | ~15k unique texts |

**Class weights** (inverse-frequency, 3-class):

| Class | Weight |
|---|---|
| supported (39.0%) | 0.91 |
| refuted (61.4%) | 0.58 |
| not_enough_evidence (6.6%) | **5.40** |

---

## 8. Module Structure

```
src/core/gnn/
├── __init__.py
├── types.py           — NodeType, EdgeType enums; VERDICT_TO_INT; ClaimGraph dataclass
├── featurizer.py      — Featurizer: encode_texts(), encode_modality(), encode_pramana()
├── graph_builder.py   — ClaimGraphBuilder.build(record) → ClaimGraph
├── dataset.py         — EpistemicFactDataset(InMemoryDataset)
├── model.py           — EpistemicHGNN (HeteroConv + GATConv)
└── train.py           — Trainer, TrainConfig, EpochResult

src/cli/
├── build_graph_dataset.py   — `just build-graph`
├── split_dataset.py         — `just split`
└── train_gnn.py             — `just train`
```

---

## 9. Training Configuration (defaults)

| Hyperparameter | Default | Notes |
|---|---|---|
| Epochs | 50 | with early stopping |
| Learning rate | 1e-3 | Adam |
| Batch size | 32 | |
| Hidden dim | 256 | all node types projected here |
| Attention heads | 2 | concat=False |
| Dropout | 0.3 | after each conv layer + in classifier |
| LR scheduler | ReduceLROnPlateau | patience=5, factor=0.5, mode=max |
| Early stopping patience | 10 | epochs without val improvement |
| Device | cpu | override with --device mps / cuda |
| Class weights | enabled | disable with --no-class-weights |

---

## 10. Running the Pipeline

```bash
# Full Phase 4 pipeline (in order):
just filter        # → out/training/epistemic_factkg_training.jsonl
just build-graph   # → out/graphs/graph_dataset.pt  (slow: embeds ~5k records)
just split         # → out/splits/{train,val,test}_indices.json
just train         # → out/checkpoints/best_model.pt

# Pathway B (modality-learned epistemic reasoning):
just train -- --use-modality-learning --aux-loss-weight 0.1

# Run tests:
uv run pytest tests/test_graph_builder.py tests/test_gnn_model.py

# Force rebuild if JSONL changes:
just build-graph -- --force-rebuild
```

---

## 11. Known Design Constraints

**Graph size variation:** Evidence cardinality varies (1–12+ items per claim). PyG's
`DataLoader` batches variable-size graphs by disjoint union — no padding required.

**Empty edge types:** All 8 edge types are always present in every `HeteroData` object.
Missing edges use `edge_index = torch.zeros([2, 0])`. This is required for PyG's
`InMemoryDataset` to produce consistent N+1 slice tensors across the collated batch.
Without this, graphs missing sparse edge types (e.g. AVeriTeC records without `has_triple`)
cause `IndexError` in `separate.py` at the first gap graph.

**String metadata:** `pramana` and `dataset` strings from Phase 3 annotations are stored
on `ClaimGraph` but not on `HeteroData` (to avoid PyG slice issues with non-tensor
attributes). They are accessible via `ClaimGraph.pramana` / `ClaimGraph.dataset`
at build time; Phase 6 evaluation will access them by re-loading the JSONL.

**Embedding cache:** The cache is keyed by 16-char SHA-256 prefix. Collision probability
is negligible for ~15k texts but the cache is not collision-safe by design.

---

## 12. Phase 5 Gates

Before Phase 5 (Training Ablations) can begin:

- [ ] `out/splits/` committed — all ablation runs must use the same split files to prevent
  data leakage
- [ ] Pathway A baseline run completed (`just train`) — establishes the accuracy floor
- [ ] Pathway B infrastructure tested (`--use-modality-learning --aux-loss-weight 0.1`)
- [ ] `best_model.pt` checkpoint loadable via `Trainer.load_best()`

**Phase 6 accommodation already in place:**  
- `get_class_weights()` returns inverse-frequency weights ready for weighted loss
- `pramana` metadata accessible for per-Pramana accuracy breakdown  
- GATConv attention coefficients extractable per edge type for explainability analysis

---

## 13. Training Results — Phase 4 Pathway A Baseline

### Observed results

| Epoch | train_loss | train_acc | val_loss | val_acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.2388     | 0.9355    | 0.0      | 1.0000  |
| 4     | 0.0462     | 0.9764    | 0.0      | 1.0000  |
| 8     | 0.0324     | 0.9756    | 0.0      | 1.0000  |
| 11    | 0.0447     | 0.9717    | 0.0      | 1.0000  |

Training stopped at epoch 11 (early stopping, patience=10). Best checkpoint: epoch 4
(train_acc=0.9764). val_acc=1.0 from the very first epoch.

### Interpretation: stance-verdict determinism

val_acc=1.0 from epoch 1 is not overfitting — it reflects a deterministic routing rule
encoded in the stance edge types:

```
any no_evidence edge  →  not_enough_evidence  (verdict=2)
any refutes edge      →  refuted              (verdict=1)
else                  →  supported            (verdict=0)
```

This rule achieves 100% accuracy on all 5,135 training records. The GNN does not need to
read claim text, sentence embeddings, or the epistemic node — the edge type pattern alone
determines the verdict in the current dataset.

### Research implication

The epistemic hypothesis ("Pramana-aware structure improves fact-verification") cannot be
tested with stance edges present. The routing shortcut dominates completely. Phase 5 must
include a **stance-removal ablation** to isolate the epistemic contribution:

| Run | Stance edges | Epistemic node | Tests |
|-----|--------------|----------------|-------|
| A | removed | absent | random-like floor (~33%) |
| B | removed | present | Pramana contribution above random |
| C | present (Pathway A) | present | stance-routing ceiling (~100%) |
| D | present (Pathway B) | learned from modality | learned vs. heuristic Pramana |

Run B is the primary test of the research hypothesis. If accuracy in Run B significantly
exceeds Run A, the epistemic node (and Pramana structure) carries independent signal
beyond the claim text alone.
