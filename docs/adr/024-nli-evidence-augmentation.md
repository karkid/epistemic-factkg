# ADR-024: NLIHybridHGNN (v3-nli) — NLI Probs as Features + Direct EC Signal

**Status:** Accepted  
**Date:** 2026-05-17  
**Builds on:** ADR-023 (HybridHGNN v2-hgnn)

---

## Context

H1 (StanceHead) predicts evidence stance from `ev_emb` — the GNN-encoded evidence
node embedding. In v2-hgnn, `ev_emb` only receives claim information indirectly
through graph message-passing over the `has_evidence` / `connected_to` edges.
This indirect path is insufficient for arbitrary natural-language stance detection:
H1 was trained on AVeriTeC/AI2THOR/synthetic text and does not generalise to
unseen domain pairs.

**Failure mode observed:** claim "Apple is red" + evidence "Apple is yellow" →
v2-hgnn predicts SUPPORTED, because the high lexical similarity (both about
apples and colours) dominates the GNN encoding and H1 cannot detect the
semantic contradiction.

A pre-trained NLI cross-encoder directly compares a (claim, evidence) text pair
and produces calibrated stance probabilities (contradiction / entailment / neutral),
generalising to any domain. The question is how to incorporate this signal.

**Options evaluated:**

| Option | Description | Verdict |
|--------|-------------|---------|
| A | NLI probs directly in EC formula (bypass H1) | Loses H1 stance supervision signal |
| B | Append NLI probs as input features only (403d) | H1 still mediates EC — doesn't generalise |
| C | Train H1 jointly with NLI cross-encoder (fine-tune) | Adds 184M parameters; out of scope |
| **A+B** | **NLI probs as features AND directly in EC formula** | **Adopted — see below** |

Option B was implemented first. Evaluation revealed it was insufficient: H1 never
learned to map NLI contradiction signal to a refutes prediction because the training
data does not cover the contradiction patterns the NLI model detects (e.g. simple
property-value contradictions like "apple is red / apple is yellow"). The NLI signal
was being diluted through the GNN's 403d→256d projection before reaching H1.

**Root cause confirmed experimentally:** the NLI cross-encoder outputs
`p_contradiction ≈ 0.998` for ("An apple is red", "An apple is yellow"), yet
v3-nli (Option B only) still predicted SUPPORTED. The NLI signal was present in
the features but H1 had not learned to use it for the EC formula.

Final decision: combine A+B — NLI probs are appended to ev_features (for the GNN
encoder) AND fed directly to the EC formula in `_soft_verdict_logits`, bypassing H1.

---

## Decision

Two changes together constitute v3-nli:

**Part 1 — NLI probs as evidence node features (403d):**

```
evidence features: [text_emb(384) | modality(8) | evidence_type(5) | source_type(3)]
                   = 400d   (GraphConfig.v1)

               + [p_contradiction | p_entailment | p_neutral]   ← NLI cross-encoder
                   = 403d   (GraphConfig.v2)
```

The NLI probs are stored in `ev.x[:, -3:]` at graph-build time. The GNN encoder
can learn to propagate this signal, but this alone is insufficient (see context above).

**Part 2 — NLI probs bypass H1 in the EC formula:**

`NLIHybridHGNN` overrides `_soft_verdict_logits` to read `ev.x[:, -3:]` directly
and feed it to `SymbolicAggregator`, skipping H1's stance prediction entirely for
the verdict pathway. The MNLI label order `[contradiction, entailment, neutral]` is
reordered to match the EC formula's convention `[supports, refutes, neutral]`:

```python
# NLIHybridHGNN._soft_verdict_logits
nli_probs    = ev.x[:, -3:]             # [p_contradiction, p_entailment, p_neutral]
stance_probs = nli_probs[:, [1, 0, 2]] # [p_supports, p_refutes, p_neutral]
# → SymbolicAggregator.compute_soft_scores(stance_probs, is_pred, ew, st)
```

H1 (`StanceHead`) still trains on its own stance CE loss and still influences the
encoder through its gradient path. It is no longer consulted for the EC formula.

**Gradient paths in v3-nli:**
```
verdict_CE → HybridVerdictHead → claim_emb   → encoder ✓  (rich 256d path)
verdict_CE → HybridVerdictHead → EC scores   → NLI probs  (frozen, no grad) —
IS_MSE     → ISHead            → encoder     ✓  (clean, detached)
stance_CE  → StanceHead        → encoder     ✓  (H1 still supervised)
```

**Academic framing:**
> "We augment evidence node representations with stance probabilities from a frozen
> NLI cross-encoder and route these probabilities directly through the EC formula,
> providing a generalised claim-evidence comparison signal that is independent of
> H1's training distribution."

---

## Alternatives Considered

**Features only (Option B alone):**
Implemented first. Insufficient — H1 does not learn to use the NLI contradiction
signal for the EC formula because the training distribution lacks the contradiction
patterns the NLI model detects. The 3 NLI dimensions are diluted in the 403d→256d
GNN projection. Superseded by the A+B combination.

**Fine-tune the NLI cross-encoder jointly:**
Adds 184M trainable parameters and significant compute. The frozen cross-encoder
already produces well-calibrated probs on out-of-domain text. Deferred as future work.

**C. Stance-typed edges instead of NLI features:**
Replace the single `has_evidence` edge type with three typed edges (supports /
refutes / neutral) derived from NLI probs. This changes the graph topology and
requires re-designing the HeteroConv message-passing. The feature-augmentation
approach achieves the same signal more simply. Deferred as future work.

---

## Results (test set, n=766 claims)

> ⚠️ Results below are from the **Option B only** version (NLI as features, H1 still
> drives EC formula). The final architecture (A+B: NLI probs directly in EC) requires
> retraining — run `just train v3-nli && just eval v3-nli` to update these numbers.

| Model    | Verdict Acc | Macro F1   | IS RMSE  | Stance Acc | averitec | synthetic |
|----------|-------------|------------|----------|------------|----------|-----------|
| baseline | 0.7950      | 0.8022     | 0.1190   | 0.7595     | 0.621    | 0.896     |
| v1-hgnn  | 0.7115      | 0.7029     | 0.1193   | 0.7488     | 0.456    | 0.896     |
| v2-hgnn  | 0.7990      | 0.8067     | 0.1161   | 0.7395     | 0.621    | 0.907     |
| v3-nli†  | 0.8146      | 0.8200     | 0.1124   | 0.7662     | 0.627    | 0.927     |

† Pre-fix results. Expected to improve further with NLI probs in EC formula.

---

## Consequences

- v3-nli is the **primary model** for publication; v2-hgnn, v1-hgnn, and baseline
  are ablations in the model-evolution table.
- Graph build time increases by ~0.3s per claim (NLI inference on CPU for N_ev pairs).
  The NLI model is loaded lazily and cached for the full dataset build.
- A separate graph dataset file (`graph_dataset_nli.pt`) is required; the 400d
  dataset (`graph_dataset.pt`) is unchanged and still used for v1-hgnn / v2-hgnn.
- At inference time (demo app), the NLI cross-encoder runs per request for v3-nli.
  Latency is acceptable for a demo (~0.2s per evidence item on CPU).
- `GraphConfig.v2()` encodes the 403d evidence dimension; all pipeline components
  (`train.py`, `evaluate.py`, `predictor.py`) select `v2()` automatically when
  `model == "v3-nli"`.
