# Experiment 01 — Shortcut Discovery

**Date:** 2026-05-12  
**Status:** Complete — negative result, led to v3.0 dataset redesign  
**ADRs:** superseded/v0 (old GNN architecture + ablation design); see docs/superseded/v0/adr/

---

## What We Tried

Built a heterogeneous GNN (`EpistemicHGNN`) to test whether making a fact-verification model aware of the *epistemic type* of evidence — using Pramana categories from Sanskrit knowledge theory — improves accuracy compared to using text embeddings alone.

**Dataset (v2.0, 2-source):** 5,135 records after filtering  
- AI2THOR: ~1,800 simulation-grounded claims (perception + non_apprehension)  
- AVeriTeC: ~3,335 web fact-check claims (testimony + inference)

**Graph schema:** Each claim → one `HeteroData` subgraph with 4 node types (claim, evidence, epistemic, triple) and 8 edge types. The 4 stance edges (`supports`, `refutes`, `absent`, `no_evidence`) are the key structural feature.

**EpistemicNode:** 6-d vector = 5-d Pramana one-hot + 1-d confidence weight. Heuristic labels from ADR-008 rules; weights from ADR-007 priors.

**Model:** 2-layer HeteroConv with GATConv per edge type → 256-d projections → Linear verdict classifier. ~2.4M parameters, 3-class (supported / refuted / not_enough_evidence).

---

## What Happened

**Phase 4 baseline (Run C): val_acc = 1.0 from epoch 1.**

This is not model success — it is a deterministic routing shortcut encoded in stance edge types:

```
any no_evidence edge  →  not_enough_evidence
any refutes edge      →  refuted
else                  →  supported
```

This rule achieves 100% accuracy on all 5,135 records. The GNN learns it at epoch 1 and never reads the claim text, sentence embeddings, or the epistemic node.

**Root cause:** The dataset labelling protocol (adapters in Phase 3) assigned both the `stance` annotation and the `verdict` label following the same rules. The labelling is internally self-consistent — there are no records where evidence `supports` the claim but the verdict is `refuted`. Stance edges therefore carry verdict information exactly.

---

## Ablation Results (Phase 5)

Ablation matrix: stance edges removed to isolate epistemic contribution.

| Run | Stance edges | Epistemic node | Test acc | Test macro F1 |
|-----|---|---|---|---|
| C — full graph (baseline) | present | present | 1.0000 | 1.0000 |
| B — no-stance, epistemic present | removed | present | 0.5725 | 0.2427 |
| A — no-stance, text only | removed | absent | 0.5725 | 0.2427 |
| D — Pathway B, modality-learned | present | learned | 1.0000 | 1.0000 |

**Run B − Run A = 0.0pp.** The epistemic hypothesis is **not supported.**

Runs A and B are identical: both collapse to predicting `refuted` (majority class, 61.4%) for every claim. The 5.4× class weight for NEE was insufficient.

**Run D** reproduces Run C's perfect accuracy because stance edges are still present — the shortcut dominates. The modality-learned Pramana pathway cannot be meaningfully tested with stance edges present.

---

## Per-Pramana Breakdown (Runs A and B — identical)

| Pramana | Test acc | n |
|---|---|---|
| `perception` | 0.797 | 128 |
| `inference` | 0.662 | 74 |
| `testimony` | 0.617 | 188 |
| `comparison_analogy` | 0.558 | 52 |
| `non_apprehension` | **0.000** | 75 |

`non_apprehension` = 0/75 correct. Without the `no_evidence` stance edge, the model has no structural signal distinguishing NEE claims. The Pramana one-hot is semantically correct but the GNN cannot extract the NEE signal from a 6-d vector under 6.6% class imbalance.

---

## Why It Failed

1. **Stance→verdict shortcut dominates.** Both real-world sources (AI2THOR, AVeriTeC) have self-consistent stance/verdict labelling. Any model will learn the routing rule before learning epistemic structure.

2. **Majority-class collapse without stance edges.** When the shortcut is removed, both models predict `refuted` for everything. NEE claims (6.6%) have no distinctive feature without the `no_evidence` edge.

3. **Pramana labels are source-identity proxies.** All AI2THOR claims → `perception`; all AVeriTeC text claims → `testimony`. The Pramana one-hot is nearly a perfect proxy for dataset origin, not genuine epistemic type.

---

## What Changed: v3.0 Dataset Redesign

The fix requires training examples where **same evidence stance + different epistemic reliability → different verdict**. This forces the GNN to learn the EC formula rather than the stance-lookup rule.

**Changes made:**
- Added a third data source: **synthetic shortcut-breaking records** (ADR-012, ADR-013)
- Moved to **per-evidence** epistemic modeling — `evidence_types`, `source_id`, `inference_strength`, `confidence_weight` on each evidence item (ADR-010)
- Added per-evidence confidence formula: $EC_i = 1 - (1-ST_i)^{EW_i \times IS_i}$ where ST comes from an external source trust registry (ADR-009)
- 15 templates generate ~62% shortcut-breaking records (supports→NEE, refutes→NEE, conflicting)
- Schema bumped to **v3.0** — `pramana_primary` abolished; per-evidence multi-label `evidence_types`

The next experiment trains the V1 neuro-symbolic GNN on the v3.0 dataset. See `docs/project-plan.md` for the implementation plan.
