# ADR-016: Phase 5 Ablation Design

## Status

Accepted

## Context

The Phase 4 Pathway A baseline reached val_acc=1.0 from epoch 1. Investigation revealed
that the four stance edge types (`supports`, `refutes`, `absent`, `no_evidence`) encode the
verdict with 100% rule accuracy:

```
any no_evidence edge  →  not_enough_evidence  (verdict=2)
any refutes edge      →  refuted              (verdict=1)
else                  →  supported            (verdict=0)
```

The GNN learns this routing rule rather than epistemic reasoning. With stance edges present,
the model does not need to read claim text, embeddings, or the epistemic node at all.

The research hypothesis — *does Pramana-aware epistemic structure improve fact verification?*
— cannot be tested under these conditions. Phase 5 removes the routing shortcut by zeroing
out stance back-edges and re-trains, producing a clean comparison between models that have
and do not have access to the EpistemicNode.

## Decision

### Ablation Matrix (4 runs)

| Run | Stance edges | Epistemic node | Run name | Primary question |
|-----|-------------|----------------|----------|------------------|
| C | present | present | `full` | Stance-routing ceiling (already done) |
| A | removed | absent | `no-stance-no-epistemic` | What does text alone achieve? (floor) |
| B | removed | present | `no-stance` | Does Pramana add signal above text? |
| D | present, modality-learned | present | `pathway-b` | Learned vs. heuristic Pramana? |

**Run C** is already complete (Phase 4 baseline). Runs A, B, D are the Phase 5 work.

**Run B is the primary test of the research hypothesis.**

### Stance-Edge Removal

"Stance edges" are the four back-edges from evidence → claim:
`supports`, `refutes`, `absent`, `no_evidence`.

These carry verdict-deterministic information. To remove them, their `edge_index` tensors
are zeroed to `[2, 0]` in each batch before the forward pass. The model architecture and
graph serialisation are unchanged — the masking happens at training time only.

The forward edge `has_evidence` (claim → evidence) is **kept**. The model can still see
that evidence nodes exist and read their embeddings; it simply loses the stance annotation.
This tests "evidence present but untagged" rather than "claim-only" — a more meaningful
comparison for the epistemic hypothesis.

For Run A, the `has_epistemic` edge (claim → epistemic node) is additionally zeroed,
removing all epistemic signal. The claim node receives only its sentence embedding and
the structure of the evidence neighbourhood.

### Implementation in Code (Phase 5)

**`src/core/gnn/train.py`** — add `masked_edge_types: list[str]` to `TrainConfig`.
In `Trainer._run_epoch()`, before the forward pass:
```python
for rel in self.config.masked_edge_types:
    for et in batch.edge_types:
        if et[1] == rel:
            batch[et].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
```

**`src/cli/train_gnn.py`** — add:
- `--no-stance-edges`: sets `masked_edge_types = ["supports", "refutes", "absent", "no_evidence"]`
- `--no-epistemic`: additionally adds `"has_epistemic"` to `masked_edge_types`
- `--run-name NAME`: writes checkpoint/history to `out/checkpoints/<NAME>/`

**CLI invocation for each run:**
```bash
# Run B: stance removed, epistemic present (primary test)
uv run python src/cli/train_gnn.py \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --no-stance-edges --run-name no-stance

# Run A: stance removed, epistemic removed (floor)
uv run python src/cli/train_gnn.py \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --no-stance-edges --no-epistemic --run-name no-stance-no-epistemic

# Run D: Pathway B — modality-learned Pramana
uv run python src/cli/train_gnn.py \
    --dataset out/graphs/graph_dataset.pt \
    --jsonl out/training/epistemic_factkg_training.jsonl \
    --use-modality-learning --aux-loss-weight 0.1 --run-name pathway-b
```

Pathway B also requires `pramana_y` labels in each graph (added to `ClaimGraphBuilder.build()`
before running) and `just build-graph -- --force-rebuild` to regenerate the `.pt` file.

### Success Criterion

**The epistemic hypothesis is supported if:**
> Run B test macro F1 − Run A test macro F1 > 5 percentage points

A 5pp threshold accounts for expected variation in a 517-record, 3-class test set.
Both runs must be evaluated on the held-out test set (not the val set used for checkpoint
selection) to prevent optimistic estimates.

### All Runs Use the Same Split Files

`out/splits/{train,val,test}_indices.json` were generated once with seed=42 in Phase 4.
All 4 ablation runs train on the same 4,106 graphs, validate on the same 512, and will be
evaluated on the same 517. This prevents data leakage between experimental conditions.

## Consequences

**Why the routing shortcut is dataset-specific, not a GNN flaw:**
In a dataset where stance labels are provided as input features, any model (GNN or otherwise)
will learn to use them because they are perfectly predictive. The shortcut is not a bug —
it reveals that stance annotations make fact verification trivially solvable. Future datasets
targeting epistemic reasoning should not include direct stance labels as model inputs, or
should require multi-hop reasoning where stance alone is insufficient.

**Why Run D is expected to show the same ceiling as Run C:**
Run D (Pathway B) still has all 4 stance edge types present — it only adds an auxiliary
Pramana prediction head. The routing shortcut is still available. Run D's purpose is to
test whether the model can *learn* Pramana labels from modality patterns (as a secondary
task) rather than to test whether Pramana improves accuracy. Comparing Run D's Pramana
head accuracy against the ADR-007 heuristic rules answers the Phase 6 question: does the
model discover the same patterns as the hand-written rules, or different ones?

**Why `absent` is a stance type, not a verdict class:**
`absent` means "AI2THOR simulation confirmed the object is physically absent" → the claim
"There is no X in this scene" is `supported`. This is different from `no_evidence` (web
search found no answer → `not_enough_evidence`). Both are stance edge types, not verdict
classes. The verdict classes (supported, refuted, not_enough_evidence) are the model's
output labels, defined in ADR-015.
