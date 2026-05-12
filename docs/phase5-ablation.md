# Phase 5: Training Ablations

**Status:** Planned  
**Date:** 2026-05-12  
**ADR:** 016

---

## Overview

Phase 5 runs three new training configurations (Runs A, B, D) to test the epistemic
hypothesis. Run C (Pathway A baseline) was completed in Phase 4 and is included for
comparison. The primary question:

> Does the EpistemicNode (Pramana type + confidence weight) add independent predictive
> signal when stance edges are removed?

---

## Background: The Routing Shortcut

Phase 4 revealed that the four stance edge types (`supports`, `refutes`, `absent`,
`no_evidence`) deterministically encode the verdict. The GNN learns:

```
any no_evidence edge  →  not_enough_evidence
any refutes edge      →  refuted
else                  →  supported
```

This rule achieves 100% accuracy. Phase 5 removes the routing shortcut by zeroing these
four back-edges and retraining — the model must then predict from claim text and
(optionally) the epistemic node alone.

See ADR-016 for the full rationale.

---

## Ablation Matrix

| Run | Stance edges | Epistemic | Run name | What it tests |
|-----|-------------|-----------|----------|---------------|
| C | present | present | `full` | Routing ceiling — **already complete** |
| A | removed | absent | `no-stance-no-epistemic` | Text-only floor |
| B | removed | present | `no-stance` | **Pramana contribution above text** |
| D | present + modality-learned | present | `pathway-b` | Learned vs. heuristic Pramana |

---

## Code Changes Required Before Running

### 1. `src/core/gnn/train.py`

Add `masked_edge_types` to `TrainConfig`:

```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    ...
    masked_edge_types: list[str] = field(default_factory=list)
```

Add masking in `_run_epoch()` before the forward pass:

```python
for rel in self.config.masked_edge_types:
    for et in batch.edge_types:
        if et[1] == rel:
            batch[et].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )
```

Update the Pathway B pramana label reference (line ~87):

```python
# before:  pramana_labels = batch.pramana_label
# after:
pramana_labels = batch["claim"].pramana_y
```

### 2. `src/core/gnn/graph_builder.py`

Add `pramana_y` to each built graph (required for Pathway B):

```python
from src.core.gnn.types import PRAMANA_TO_INT

# in ClaimGraphBuilder.build(), after setting data["claim"].x:
pramana_idx = PRAMANA_TO_INT.get(pramana_primary, -1)
data["claim"].pramana_y = torch.tensor([pramana_idx], dtype=torch.long)
```

After this change, regenerate the dataset:

```bash
just build-graph -- --force-rebuild
```

### 3. `src/cli/train_gnn.py`

Add three new flags:

```python
ap.add_argument(
    "--no-stance-edges",
    action="store_true",
    help="Remove stance back-edges (supports/refutes/absent/no_evidence) — Run A/B",
)
ap.add_argument(
    "--no-epistemic",
    action="store_true",
    help="Also remove has_epistemic edge — Run A only",
)
ap.add_argument(
    "--run-name",
    default=None,
    help="Checkpoint sub-directory name (e.g. 'no-stance')",
)
```

Wire them into config and checkpoint_dir:

```python
masked = []
if args.no_stance_edges:
    masked = ["supports", "refutes", "absent", "no_evidence"]
if args.no_epistemic:
    masked.append("has_epistemic")

ckpt_dir = args.checkpoint_dir
if args.run_name:
    ckpt_dir = str(Path(args.checkpoint_dir) / args.run_name)

config = TrainConfig(
    ...,
    masked_edge_types=masked,
    checkpoint_dir=ckpt_dir,
)
```

### 4. `Justfile`

Add `ablation` target:

```makefile
ablation: build-graph split
    @echo "=== Run B: no-stance, epistemic present ==="
    uv run python src/cli/train_gnn.py \
        --dataset out/graphs/graph_dataset.pt \
        --jsonl out/training/epistemic_factkg_training.jsonl \
        --no-stance-edges --run-name no-stance --epochs 50

    @echo "=== Run A: no-stance, no epistemic ==="
    uv run python src/cli/train_gnn.py \
        --dataset out/graphs/graph_dataset.pt \
        --jsonl out/training/epistemic_factkg_training.jsonl \
        --no-stance-edges --no-epistemic --run-name no-stance-no-epistemic --epochs 50

    @echo "=== Run D: Pathway B (modality-learned) ==="
    uv run python src/cli/train_gnn.py \
        --dataset out/graphs/graph_dataset.pt \
        --jsonl out/training/epistemic_factkg_training.jsonl \
        --use-modality-learning --aux-loss-weight 0.1 --run-name pathway-b --epochs 50
```

---

## Running the Ablations

```bash
# Prerequisites
just build-graph -- --force-rebuild   # adds pramana_y to graphs
just split                             # already done; verify files exist

# Run all three new configurations
just ablation

# Or run individually:
just train -- --no-stance-edges --run-name no-stance
just train -- --no-stance-edges --no-epistemic --run-name no-stance-no-epistemic
just train -- --use-modality-learning --aux-loss-weight 0.1 --run-name pathway-b
```

---

## Expected Results

| Run | Expected val_acc | Expected test macro F1 | Notes |
|-----|-----------------|----------------------|-------|
| C (full graph) | ~1.00 | ~1.00 | Phase 4 confirmed |
| B (no-stance, epist.) | 0.50–0.75 | 0.40–0.65 | **Hypothesis primary test** |
| A (no-stance, no-epist.) | 0.45–0.65 | 0.35–0.55 | Text-only floor |
| D (Pathway B) | ~1.00 | ~1.00 | Stance still present |

**Hypothesis supported if Run B macro F1 > Run A macro F1 by > 5pp** on test set.
See ADR-016 for the full success criterion.

---

## Checkpoint Layout After Phase 5

```
out/checkpoints/
├── best_model.pt                    ← Run C (Phase 4 full baseline)
├── training_history.json            ← Run C history
├── no-stance/
│   ├── best_model.pt                ← Run B
│   └── training_history.json
├── no-stance-no-epistemic/
│   ├── best_model.pt                ← Run A
│   └── training_history.json
└── pathway-b/
    ├── best_model.pt                ← Run D
    └── training_history.json
```

---

## Tests

`tests/test_trainer.py` (new):

| Test | What it verifies |
|------|-----------------|
| `test_masked_edge_types_zeroed` | Specified edge types have empty edge_index after masking |
| `test_non_masked_edges_unchanged` | Unmasked edge types are not affected |
| `test_fit_saves_checkpoint` | Training loop writes best_model.pt |
| `test_early_stopping` | Training stops before max_epochs when val does not improve |
| `test_load_best_restores_weights` | load_best() restores checkpoint weights |

---

## Phase 5 Gates (before Phase 6)

- [ ] `just build-graph -- --force-rebuild` completed — `pramana_y` in all graphs
- [ ] `just ablation` completed — all 3 new checkpoint directories exist
- [ ] All 4 training histories committed to `out/checkpoints/`
- [ ] `pytest tests/test_trainer.py` passes
