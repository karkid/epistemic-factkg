# ADR-026: Encoder Residual Connections and Windowed Co-Evidence

**Status:** Accepted  
**Date:** 2026-05-18  
**Builds on:** ADR-023 (HybridHGNN), ADR-024 (NLIHybridHGNN)

---

## Context

The shared `EpistemicEncoder` is a two-layer HeteroConv GNN with GATConv message
passing. Two problems were identified after v3-nli training:

### Problem 1 — NLI feature dilution through the encoder

v3-nli appends 3 NLI probability dimensions to the 400-dim evidence features (ADR-024),
producing 403-dim input nodes. The NLI probs encode precise stance information
that the EC formula depends on. Without residual connections, the GATConv projection
(`403d → 256d`) has no mechanism to preserve the original NLI signal — the
cross-encoder's calibrated probabilities are aggregated together with all other
features in the attention mechanism, and the gradient signal from the verdict loss
cannot easily reconstruct them.

### Problem 2 — Oversmoothing in dense AVeriTeC co-evidence graphs

The co-evidence edge set connected every evidence node to every other evidence node
in the same claim (`O(N²)` edges). For sparse AI2THOR and synthetic claims (typically
2–4 evidence items) this was fine. But AVeriTeC claims have up to 16 evidence items,
producing up to `16 × 15 = 240` co-evidence edges per claim.

Two GATConv layers over a dense co-evidence graph cause **oversmoothing**: all evidence
embeddings converge to nearly identical representations (averaged over all pairwise
neighbours). When all evidence nodes look alike, H1 (StanceHead) and the EC formula
lose the ability to differentiate supporting from refuting evidence within the same
claim.

This also explains why v1-hgnn had worse AVeriTeC accuracy despite having the EC
formula — the EC formula needs distinguishable per-evidence stance signals.

---

## Decision

Two complementary fixes, both applied to `EpistemicEncoder` and `ClaimGraphBuilder`.

### Fix 1 — Residual skip connections at each encoder layer

After each HeteroConv layer, add the pre-layer activation to the layer output:

```python
# Layer 1
x0 = {ntype: act(proj(x)) for ntype, proj in input_proj.items()}
x1 = conv1(x0, edge_index_dict)
x1 = {k: act(dropout(v)) for k, v in x1.items()}
x1 = {k: x1[k] + x0[k] for k in x1}     # ← residual

# Layer 2
x2 = conv2(x1, edge_index_dict)
x2 = {k: act(v) for k, v in x2.items()}
x2 = {k: x2[k] + x1[k] for k in x2}     # ← residual
```

Dimension compatibility is guaranteed: GATConv with `heads=4` and
`out_channels = hidden_dim // heads` produces output of `hidden_dim // heads * heads
= hidden_dim`. The residual is added without a learned projection.

The residual paths preserve the input projection (including NLI probs for v3-nli)
in the final embedding, giving H1 and the EC formula direct access to the original
feature signal alongside the message-passed context.

### Fix 2 — Windowed co-evidence (max-5 nearest neighbours)

Replace the fully-connected co-evidence graph with a windowed neighbourhood:
for each evidence node `i`, connect only to the 5 nearest nodes by index distance
`|i − j|`.

```python
_MAX_CO_EV = 5
for i in range(n_ev):
    near = sorted([j for j in range(n_ev) if j != i], key=lambda j: abs(j - i))
    for j in near[:_MAX_CO_EV]:
        pairs.append((i, j))
```

**Why index proximity?** Evidence items are ordered by their position in the source
document or Q+A chain. In AVeriTeC, consecutive Q+A pairs address related sub-questions
about the same claim aspect. In synthetic data, evidence items are ordered by
inference chain. Index proximity is therefore a meaningful similarity proxy that
preserves local context without the oversmoothing cost of full connectivity.

Max-5 bounds the co-evidence edge count at `5 × N_ev`, linear rather than quadratic.
For the max AVeriTeC case (16 evidence items), this drops from 240 to 80 edges.

---

## Alternatives Considered

**A. Symmetric k-NN by embedding similarity:** Connect each evidence node to its
k-most-similar neighbours by cosine distance. More semantically accurate but requires
a pre-pass over embeddings, adding significant build time. Index proximity is a good
structural proxy for short evidence sequences. Deferred.

**B. Stance-typed co-evidence edges:** Three separate edge types (support–support,
support–refute, etc.) to let the GNN learn different message-passing for different
co-evidence relationships. Requires stance labels at build time (pre-model). Deferred.

**C. Larger hidden dim instead of residuals:** Widen the GNN to 512d so information
capacity is larger. Doubles parameter count with no interpretability benefit. Rejected.

**D. Single encoder layer:** Remove the second GATConv layer to reduce oversmoothing.
Reduces representational power; two layers allow the encoder to propagate claim
context to evidence (layer 1: claim→evidence via `has_evidence`) and then back
(layer 2: evidence→claim via `connected_to`). Rejected.

---

## Consequences

- **No new parameters** — residual connections are parameter-free.
- **Applicable to all three HGNN models** (v1-hgnn, v2-hgnn, v3-nli) since they
  share `EpistemicEncoder`. All models benefit from the residual + windowed graph.
- **_PROJ_DIM constraint:** The residual connections preserve claim text embeddings
  strongly in the final `claim_emb`. This makes the `HybridVerdictHead`'s `claim_proj`
  more expressive. `_PROJ_DIM = 16` (4,096 params in `claim_proj`) is required;
  `_PROJ_DIM = 64` causes the head to memorise training claims and fails to generalise
  (observed val-test gap of 28.6pp during experimentation).
- Graph build is unchanged — the windowed co-evidence change is in `ClaimGraphBuilder`,
  not in the encoder. Existing graph dataset files must be rebuilt (`just graph` /
  `just graph-nli`) for the new edge structure to take effect.
