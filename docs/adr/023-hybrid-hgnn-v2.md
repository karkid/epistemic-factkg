# ADR-023: HybridHGNN (v2-hgnn) — Fused EC Scores and Claim Embedding

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-014 (VerdictHead), ADR-016 (baseline), ADR-022 (IS detach)

---

## Context

After applying IS detach (ADR-022), IS prediction quality improved significantly
(RMSE 0.12) but verdict accuracy dropped (v1-hgnn: 71.1% vs baseline: 79.5%).
Profiling the failure revealed a **2D information bottleneck**:

```
v1-hgnn verdict path:  EC scores [2d] → Linear(2→3) → verdict
baseline verdict path:  claim_emb [256d] → MLP → verdict
```

With IS detached, the encoder can only receive verdict gradient through
`stance_probs` (via `EC × p_stance`). The VerdictHead's 2D input cannot express
the full diversity of claim semantics, and the encoder is under-supervised for
the verdict task.

**Key diagnostic:** both v1-hgnn and baseline achieved identical synthetic accuracy
(89.6%) — the EC formula provided zero additional benefit on its own domain, meaning
VerdictHead was not learning to exploit EC scores.

---

## Decision

Create `HybridHGNN` (registered as `"v2-hgnn"`) with a `HybridVerdictHead` that
concatenates the symbolic EC scores with the claim node embedding:

```python
# HybridVerdictHead.forward
combined = torch.cat([scores, claim_emb], dim=1)  # [N_claims, hidden_dim + 2]
return self.mlp(combined)                          # Linear(258→128) → ReLU → Linear(128→3)
```

Everything else is identical to EpistemicHGNN: same encoder, StanceHead, ISHead,
EC formula, SymbolicAggregator, and IS detach.

Gradient paths in v2-hgnn:

```
verdict_CE → HybridVerdictHead → claim_emb  → encoder   (rich 256d path) ✓
verdict_CE → HybridVerdictHead → EC scores  → stance     → encoder        ✓
IS_MSE     → is_pred           → ISHead     → encoder   (clean, detached) ✓
```

---

## Alternatives Considered

**A. Revert IS detach** — restores verdict accuracy through gradient coupling but
IS RMSE returns to ~0.23, undermining epistemic interpretability. Rejected.

**B. Larger VerdictHead on 2D EC scores** — deeper MLP on 2D input cannot recover
information lost in the symbolic aggregation. Rejected.

**C. Attention over evidence embeddings for verdict** — replaces the symbolic
aggregation with learned attention. Loses the interpretable EC formula entirely.
Saved as a future direction; the hybrid preserves interpretability.

---

## Results (test set)

| Model    | Verdict Acc | Macro F1 | IS RMSE | averitec | synthetic |
|----------|-------------|----------|---------|----------|-----------|
| baseline | 0.7950      | 0.8022   | 0.1190  | 0.621    | 0.896     |
| v1-hgnn  | 0.7115      | 0.7029   | 0.1193  | 0.456    | 0.896     |
| v2-hgnn  | **0.7990**  | **0.8067** | **0.1161** | **0.621** | **0.907** |

v2-hgnn outperforms baseline on verdict Macro F1 (+0.45pp) with identical IS quality.
The EC formula contributes a small but real advantage on synthetic data (+1.1pp) where
epistemic consistency holds. On AVeriTeC, v2-hgnn matches baseline — the source-trust
mismatch in crowdsourced annotations limits the EC formula's advantage on real data.

---

## Consequences

- v2-hgnn is the **primary model** for publication; v1-hgnn and baseline are ablations.
- The HybridVerdictHead adds `hidden_dim × hidden_dim/2 + hidden_dim/2 × 3` parameters
  (~33k at hidden_dim=256) — negligible relative to the encoder.
- IS interpretability is maintained: IS RMSE 0.116 means the model's IS predictions
  are accurate enough to discuss in the paper.
- The 2D EC scores (`support_score`, `refute_score`) remain interpretable intermediate
  outputs, now paired with the claim embedding for final verdict classification.
