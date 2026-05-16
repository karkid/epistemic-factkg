# ADR-016: BaselineHGNN — Neural Ablation Without EC Formula

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-013 (EpistemicHGNN architecture), ADR-014 (VerdictHead)

---

## Context

EpistemicHGNN's central contribution is the neuro-symbolic EC formula pathway:

```
evidence embeddings → IS → EC formula → SymbolicAggregator → VerdictHead → verdict
```

Reviewers will ask: *does the EC formula actually help, or does the improvement come
purely from the graph encoder?* Without an ablation that removes the formula while
keeping everything else constant, this question cannot be answered.

---

## Decision

Create `BaselineHGNN` — identical encoder, StanceHead, and ISHead as EpistemicHGNN,
but the verdict is predicted directly from the **claim node embedding** via an MLP:

```
claim_emb [hidden_dim] → Linear → ReLU → Dropout → Linear → verdict logits [3]
```

The EC formula, SymbolicAggregator, and VerdictHead are absent.

Everything upstream of verdict is shared: same graph structure, same HeteroConv
encoder, same multi-task loss (stance CE + IS MSE + verdict CE).

The model is registered as `"baseline"` in the model registry.

---

## Alternatives Considered

**A. Zero-out EC formula at inference only** — does not cleanly isolate gradient
effects during training. Rejected in favour of a structurally distinct model.

**B. Random-weight encoder** — tests whether the encoder itself adds value, but
cannot answer the EC formula question. A separate ablation if needed.

---

## Consequences

- Provides the necessary ablation for publication: *EC formula vs no EC formula,
  all else equal.*
- IS head is present in baseline but its output is not used in verdict — IS
  regression runs as an auxiliary task to keep the training objective comparable.
- BaselineHGNN IS RMSE is consistently better than EpistemicHGNN (0.10 vs 0.23
  in early runs) because IS is supervised without verdict gradient interference —
  a finding that motivated ADR-022.
