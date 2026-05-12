# ADR-013: GNN Architecture — HeteroConv + GATConv

## Status

Accepted

## Context

Phase 4 requires a GNN that classifies claims into 3 verdict labels (supported, refuted, not_enough_evidence) using a heterogeneous per-claim subgraph. The graph has 4 node types (claim, evidence, epistemic, triple) and 7 edge types with different semantic meanings. The architecture must:

1. Process structurally incompatible inputs: AI2THOR produces structured triples; AVeriTeC produces text sentences
2. Incorporate `confidence_weight` as a prior on the epistemic edge (ADR-004, ADR-008)
3. Produce per-evidence attention weights for explainability analysis in Phase 6
4. Train on ~5,000 records — a small dataset by GNN standards

Options considered:

| Architecture | Attention / explainability | `edge_attr` support | Parameters at ~5k records | Notes |
|---|---|---|---|---|
| **HeteroConv + GATConv** | ✓ per-edge coefficients | ✓ native | Moderate | Selected |
| HeteroConv + SAGEConv | ✗ mean aggregation | ✗ | Low | Cannot pass confidence_weight as edge prior |
| HGT (Heterogeneous Graph Transformer) | ✓ type-specific attention | ✓ | High | Too many parameters; overfitting risk at 5k |
| RGCN (Relational GCN) | ✗ | ✗ | Low | No attention; explainability not possible |

## Decision

Use **HeteroConv wrapping GATConv per edge type, 2 layers, 2 attention heads, hidden_dim=256**.

```
Input node features (variable dim per type):
  claim:     384-d  (sentence-transformer embedding)
  evidence:  389-d  (embedding + modality one-hot 5-d)
  epistemic: 6-d    (pramana_primary one-hot 5-d + confidence_weight 1-d)
  triple:    384-d  (AI2THOR only — "s p o" text → embedding)

Linear projection: each node type → hidden_dim=256

Layer 1: HeteroConv(aggr='mean')
  ├── (claim, has_evidence, evidence)   : GATConv(heads=2, concat=False)
  ├── (evidence, supports, claim)       : GATConv(heads=2, concat=False)
  ├── (evidence, refutes, claim)        : GATConv(heads=2, concat=False)
  ├── (evidence, absent, claim)         : GATConv(heads=2, concat=False)
  ├── (claim, has_epistemic, epistemic) : GATConv(heads=2, concat=False, edge_dim=1)
  ├── (claim, has_triple, triple)       : GATConv(heads=2, concat=False)  [AI2THOR only]
  └── (evidence, from_triple, triple)   : GATConv(heads=2, concat=False)  [AI2THOR only]

ReLU → Dropout(0.3)
Layer 2: same HeteroConv structure
ReLU → Dropout(0.3)
Verdict classifier: Linear(256 → 3)  [claim node embedding only]

[Pathway B — Phase 5 ablation]
Epistemic aux head: Linear(256 → 5)  [epistemic node → Pramana prediction]
```

## Consequences

**Why GATConv and not SAGEConv:**
SAGEConv aggregates by mean/max with no attention and no `edge_attr` API. It cannot pass `confidence_weight` as an edge prior — the Pramana weight would need to be moved to the node feature, losing its relational meaning (ADR-014). Critically, SAGEConv produces no per-edge attention coefficients, making Phase 6 per-evidence explainability impossible.

**Why GATConv and not HGT:**
HGT uses type-specific projection matrices for each (source_type, edge_type, target_type) combination. With 7 edge types and 4 node types this multiplies parameters significantly. At ~5k training records this creates a high overfitting risk. GATConv is expressive enough for this dataset size.

**Why GATConv and not RGCN:**
RGCN uses one weight matrix per relation type without attention. Same explainability problem as SAGEConv.

**Why 2 layers:**
Layer 1 captures direct claim↔evidence and claim↔epistemic relationships (1-hop). Layer 2 captures claim→evidence→triple paths for AI2THOR records (2-hop). More than 2 layers causes over-smoothing on small subgraphs (average ~6–8 nodes per subgraph).

**Why heads=2, concat=False:**
The dataset is too small for 8-head attention (standard for large graphs). `concat=False` averages across heads, keeping the output at 256-d rather than 512-d, controlling parameter count.

**Why per-claim subgraphs (not one big graph):**
No meaningful cross-claim edges exist — claims are independent. A single large graph would introduce false structural connections between unrelated claims and make batching impractical. Per-claim PyG `HeteroData` objects are the standard pattern for claim verification tasks.

**Pathway A / Pathway B switch:**
A `use_modality_learning: bool` config flag distinguishes Pathway A (heuristic `pramana_primary` as fixed input) from Pathway B (Pramana deduced from evidence modalities, with an auxiliary prediction head). The architecture shape is identical — only the epistemic node input and the aux head activation differ. This is a Phase 5 ablation; Phase 4 trains Pathway A only.
