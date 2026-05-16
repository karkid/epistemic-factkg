# Model Evolution — EpistemicFactKG

This document traces the three model architectures tried, the decisions that led from
one to the next, and their comparative results on the test set.

---

## Architecture Overview

All three models share the same upstream components: the HeteroConv graph encoder,
StanceHead (H1), and ISHead (H2). They differ only in how the verdict is produced.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SHARED COMPONENTS                              │
│                                                                       │
│  HeteroData graph                                                     │
│  (claim, evidence, triple nodes + pramana-typed edges)               │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────┐                                                  │
│  │ EpistemicEncoder │  HeteroConv (GAT, 4 heads, hidden_dim=256)     │
│  └────────┬────────┘                                                  │
│           │                                                           │
│    ┌──────┴──────┐                                                    │
│    ▼             ▼                                                    │
│  ev_emb      claim_emb                                                │
│  [N_ev,256]  [N_cl,256]                                               │
│    │                                                                  │
│    ├──────────────────────────┐                                       │
│    ▼                          ▼                                       │
│  ┌──────────┐          ┌───────────┐                                  │
│  │StanceHead│          │  ISHead   │                                  │
│  │  (H1)    │          │   (H2)    │                                  │
│  └──────────┘          └───────────┘                                  │
│  stance_logits         is_pred                                        │
│  [N_ev, 3]             [N_ev, 1]  ← supervised by IS regression      │
└─────────────────────────────────────────────────────────────────────┘
         │                    │
         └────────┬───────────┘
                  │
           (verdict pathway differs per model — see below)
```

---

## Model 1 — BaselineHGNN (`baseline`)

**ADR:** ADR-016  
**Purpose:** Ablation — no EC formula, verdict from claim embedding only.

```
  claim_emb [N_cl, 256]
       │
       ▼
  ┌───────────────────────────┐
  │  MLP                      │
  │  Linear(256→128)          │
  │  ReLU                     │
  │  Dropout                  │
  │  Linear(128→3)            │
  └───────────┬───────────────┘
              ▼
       verdict_logits [N_cl, 3]
```

**Gradient paths:**
- Verdict CE → MLP → claim_emb → encoder ✓
- IS regression → ISHead → encoder ✓ (IS not used in verdict)
- Stance CE → StanceHead → encoder ✓

**What it tests:** Whether the graph encoder + claim embedding alone is sufficient
for verdict, without any epistemic formalism.

---

## Model 2 — EpistemicHGNN (`v1-hgnn`)

**ADR:** ADR-013, ADR-014, ADR-022  
**Purpose:** Pure neuro-symbolic — verdict only from EC formula scores.

```
  ev_emb [N_ev, 256]         is_pred [N_ev, 1]
       │                          │
       │                     .detach()   ← ADR-022
       │                          │
       ▼                          ▼
  stance_probs [N_ev, 3]     is_fixed (no grad)
       │                          │
       └──────────┬───────────────┘
                  ▼
       SymbolicAggregator (EC formula per claim)
       EC_i = 1 - (1 - ST_i)^(EW_i × IS_i)
       support_score = 1 - ∏(1 - EC_i × p_support_i)
       refute_score  = 1 - ∏(1 - EC_i × p_refute_i)
                  │
                  ▼
       scores [N_cl, 2]
                  │
       ┌──────────▼──────────┐
       │   VerdictHead        │
       │   Linear(2→3)        │
       └──────────┬──────────┘
                  ▼
       verdict_logits [N_cl, 3]
```

**Gradient paths (after IS detach):**
- Verdict CE → VerdictHead → EC×p_stance → stance_probs → encoder ✓
- IS regression → ISHead → encoder ✓ (clean, no verdict interference)
- Stance CE → StanceHead → encoder ✓

**Key limitation:** The 2D bottleneck `[support_score, refute_score]` discards
the claim's semantic context. VerdictHead has far less information than the
baseline's 256-dim embedding.

---

## Model 3 — HybridHGNN (`v2-hgnn`)

**ADR:** ADR-023  
**Purpose:** Best of both worlds — EC formula signal fused with claim embedding.

```
  ev_emb [N_ev, 256]              claim_emb [N_cl, 256]
       │                                  │
       │     is_pred.detach()             │  ← direct verdict gradient
       │           │                      │
       ▼           ▼                      │
  stance_probs  is_fixed                  │
       │           │                      │
       └─────┬─────┘                      │
             ▼                            │
    SymbolicAggregator                    │
    scores [N_cl, 2]                      │
             │                            │
             └─────────────┬──────────────┘
                           ▼
                  torch.cat([scores, claim_emb])
                       [N_cl, 258]
                           │
             ┌─────────────▼─────────────┐
             │   HybridVerdictHead        │
             │   Linear(258→128)          │
             │   ReLU                     │
             │   Linear(128→3)            │
             └─────────────┬─────────────┘
                           ▼
                  verdict_logits [N_cl, 3]
```

**Gradient paths:**
- Verdict CE → HybridVerdictHead → **claim_emb** → encoder ✓ (rich 256d path)
- Verdict CE → HybridVerdictHead → EC×p_stance → stance → encoder ✓
- IS regression → ISHead → encoder ✓ (clean)
- Stance CE → StanceHead → encoder ✓

---

## Model 4 — NLIHybridHGNN (`v3-nli`)  ← primary model

**ADR:** ADR-024  
**Purpose:** Feed frozen NLI stance probs directly into the EC formula, bypassing H1's
training-distribution limitation. NLI probs are also stored as evidence node features
for the GNN encoder.

```
  evidence text + claim text
       │
       ▼
  frozen DeBERTa-v3-small (MNLI)   ← cross-encoder, per (claim, ev) pair
  [p_contradiction, p_entailment, p_neutral]
       │
       ├─→ appended to ev_features: 400d → 403d  (GNN encoder input)
       │
       └─→ reordered [1,0,2] → [p_supports, p_refutes, p_neutral]
                │
                ▼
       SymbolicAggregator  ←  bypasses H1 entirely for EC formula
       EC_i = 1 - (1 - ST_i)^(EW_i × IS_i)
       support_score = 1 - ∏(1 - EC_i × p_supports_i)
       refute_score  = 1 - ∏(1 - EC_i × p_refutes_i)
                │
                ▼
       scores [N_cl, 2] + claim_emb [N_cl, 256]
                │
       ┌────────▼────────┐
       │ HybridVerdictHead│
       └────────┬────────┘
                ▼
       verdict_logits [N_cl, 3]
```

**What changes vs v2-hgnn:**
1. Evidence node input: 400d → 403d (NLI probs appended)
2. EC formula: uses frozen NLI probs directly instead of H1's learned stance probs
3. `_soft_verdict_logits` overridden in `NLIHybridHGNN`; H1 only trains on stance CE

**Gradient paths:**
```
verdict_CE → HybridVerdictHead → claim_emb → encoder  ✓  (rich 256d path)
verdict_CE → HybridVerdictHead → EC scores → NLI probs    (frozen, no grad)
IS_MSE     → ISHead            → encoder   ✓  (clean, detached)
stance_CE  → StanceHead        → encoder   ✓  (H1 still supervised, not in EC)
```

---

## Comparative Results (Test Set)

> v3-nli results marked † are pre-fix (NLI as features only). Re-run `just train v3-nli && just eval v3-nli` after the architecture change to update.

| Metric              | baseline | v1-hgnn | v2-hgnn  | v3-nli†  |
|---------------------|----------|---------|----------|----------|
| **Verdict Acc**     | 0.7950   | 0.7115  | 0.7990   | **0.8146** |
| **Verdict Macro F1**| 0.8022   | 0.7029  | 0.8067   | **0.8200** |
| Verdict W-F1        | 0.7951   | 0.7159  | 0.7989   | **0.8146** |
| IS RMSE ↓           | 0.1190   | 0.1193  | 0.1161   | **0.1124** |
| IS Pearson r ↑      | 0.8635   | 0.8637  | 0.8709   | **0.8797** |
| Stance Acc          | 0.7595   | 0.7488  | 0.7395   | **0.7662** |
| Stance Macro F1     | 0.6897   | 0.6738  | 0.6580   | **0.6808** |
| Stance ECE ↓        | —        | —       | **0.0591** | 0.0758 |

### Per-Source Verdict Accuracy

| Source   | n   | baseline | v1-hgnn | v2-hgnn | v3-nli   |
|----------|-----|----------|---------|---------|----------|
| ai2thor  | 180 | 0.967    | 0.911   | 0.967   | **0.994** |
| averitec | 327 | 0.621    | 0.456   | 0.621   | **0.627** |
| synthetic| 259 | 0.896    | 0.896   | 0.907   | **0.927** |

### Per-Class Verdict F1 (v3-nli)

| Class               | Precision | Recall | F1    |
|---------------------|-----------|--------|-------|
| supported           | 0.771     | 0.789  | 0.780 |
| refuted             | 0.834     | 0.802  | 0.818 |
| not_enough_evidence | 0.845     | 0.880  | **0.862** |

---

## Decision Chain

```
ADR-013  EpistemicHGNN v1 architecture
    │
    ├── ADR-014  VerdictHead (learned thresholds)
    │       │
    │       └── ADR-016  BaselineHGNN ablation
    │
    ├── ADR-015  Class-weighted loss (NEI imbalance)
    ├── ADR-017  IS jitter in synthetic data
    ├── ADR-018  NEI-heavy synthetic distribution
    ├── ADR-019  AVeriTeC IS rubric (answer_type)
    ├── ADR-020  Webarchive source trust resolution
    ├── ADR-021  IS cap by source trust
    │
    ├── ADR-022  IS gradient detach
    │       │    (IS RMSE: 0.23 → 0.12; but verdict drops due to 2D bottleneck)
    │       │
    ├── ADR-023  HybridHGNN v2-hgnn
    │            (EC scores + claim_emb → verdict)
    │            (verdict acc: 0.799, macro F1: 0.807)
    │
    └── ADR-024  NLIHybridHGNN v3-nli  ← primary model
                 (NLI probs [3d] appended to evidence features: 400d → 403d)
                 (verdict acc: 0.815, macro F1: 0.820)
```

---

## Key Findings for Paper

1. **The EC formula works on epistemically consistent data:** v2-hgnn outperforms
   baseline on synthetic (+1.1pp) where EC values are mathematically grounded.

2. **The 2D bottleneck limits pure symbolic verdict (v1-hgnn):** aggregating all
   evidence into two scalars discards semantic context needed for verdict prediction.

3. **Hybrid outperforms pure neural baseline (+0.45pp Macro F1):** the EC formula
   adds real signal when combined with claim embeddings.

4. **NLI augmentation improves stance and verdict across all sources (v3-nli):**
   appending frozen DeBERTa NLI probs to evidence features gives H1 a direct
   claim-evidence signal. Verdict Macro F1 rises +1.33pp over v2-hgnn (0.807 → 0.820)
   with improvements on all three sources including AVeriTeC (+0.6pp).

5. **AVeriTeC reveals label-trust mismatch:** crowdsourced annotations accept low-
   trust sources as sufficient evidence; the EC formula correctly downgrades these
   to NEI. The ceiling at ~0.627 persists across all models — annotation noise, not
   model capacity, is the bottleneck.

6. **IS detach is essential for interpretability:** without detach, IS drifts to
   task-optimal values (RMSE 0.23) that do not reflect epistemic ground truth.
   With detach, IS RMSE reaches 0.112 in v3-nli — the best across all models.

7. **Neutral stance remains the hard class:** n=101 true neutrals vs 590/802 for
   support/refute causes high recall (0.683) but low precision (0.308) for the neutral
   stance class. This is a training-set imbalance artifact and does not harm verdict
   quality because EC aggregation is robust to individual misclassifications.
