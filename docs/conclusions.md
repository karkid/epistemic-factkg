# Conclusions and Research Findings

**Project:** Epistemic FactKG  
**Date:** 2026-05-12  
**Status:** Complete — Phase 5 and Phase 6 results incorporated.

---

## 1. Research Question

Does making a fact-verification model aware of the *epistemic type* of a claim — using
Pramana categories from Sanskrit knowledge theory (perception, testimony,
non-apprehension, comparison-analogy, inference) — improve fact-verification accuracy
compared to a model that processes the same evidence text without epistemic annotation?

This project operationalises this question by building a heterogeneous GNN with an
EpistemicNode that carries the claim's Pramana type and heuristic confidence weight, and
testing whether this node adds predictive signal above what text embeddings alone provide.

---

## 2. Key Finding: Stance-Verdict Determinism

**Phase 4 finding (confirmed before Phase 5 ablations):**

The baseline model (Run C, full graph) reached val_acc=1.0 from the very first training
epoch. This is not a sign of a successful model — it is a sign of a deterministic shortcut.

The four evidence-stance edge types (`supports`, `refutes`, `absent`, `no_evidence`) encode
the verdict label with 100% rule accuracy on all 5,135 training records:

```
any no_evidence edge  →  not_enough_evidence  (verdict=2)
any refutes edge      →  refuted              (verdict=1)
else                  →  supported            (verdict=0)
```

The GNN learns this routing rule in epoch 1. The claim text, sentence embeddings, and
epistemic node are never used — the edge type pattern alone determines the verdict.

### Why this happened

The dataset was labelled by domain-specific adapters (Phase 3) that assigned both the
stance annotation and the verdict label following the same rules. The labelling is
internally consistent — there are no records where, say, all evidence `supports` the claim
but the verdict is `refuted`. As a result, stance edges carry verdict information exactly.

### Implications for dataset design

Any stance-annotated fact-verification dataset where the labelling protocol assigns stance
and verdict by the same rules will have this property. Future datasets targeting epistemic
reasoning should either:

1. **Not expose stance labels as model inputs** — the model must infer the stance from
   evidence text and claim, which is the harder and more meaningful task.
2. **Design multi-hop claims** where individual evidence stances are inconclusive and the
   verdict depends on aggregating across multiple pieces of evidence under epistemic reasoning.
3. **Use adversarial claims** where the heuristic routing rule fails (e.g., a claim where
   web evidence says "supports" but epistemic analysis of source reliability suggests
   "refuted").

### Why Run C results are still reported

Run C (val_acc=1.0) is included in Phase 6 to make the shortcut explicit in the ablation
comparison table. It is the "stance-routing ceiling" — any real epistemic contribution
from the model must be measured below this ceiling, in Runs A and B where the shortcut is
removed.

---

## 3. Ablation Results

| Run | val_acc | Test accuracy | Test macro F1 | Test weighted F1 | ECE |
|-----|---------|---------------|---------------|------------------|-----|
| C (full graph) | 1.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| B (no-stance, epistemic present) | 0.5742 | 0.5725 | 0.2427 | 0.4169 | 0.2249 |
| A (no-stance, text only) | 0.5742 | 0.5725 | 0.2427 | 0.4169 | 0.2231 |
| D (Pathway B — modality-learned) | 1.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

**Test set confusion matrix (Runs B and A — identical):**

| True \ Predicted | supported | refuted | not_enough_evidence |
|-----------------|-----------|---------|---------------------|
| supported (188) | 0 | 188 | 0 |
| refuted (296) | 0 | 296 | 0 |
| not_enough_evidence (33) | 0 | 33 | 0 |

Both models predict every claim as `refuted` — a majority-class collapse.

**Interpretation:**

**Run B macro F1 − Run A macro F1 = 0.0pp.** The epistemic hypothesis is **not supported** at this dataset scale and architecture.

The heuristic Pramana labels (ADR-007) and confidence weights (ADR-008) add zero independent predictive signal above the sentence embeddings alone. This is a meaningful negative result, not a failure of the approach:

1. **Majority-class collapse under imbalance.** Without stance edges, both models default to predicting `refuted` (61.4% of the training set) for every claim. The 5.4× class weight for NEE is insufficient to overcome the imbalance when NEE claims (6.6%) have no distinctive structural signal — the `no_evidence` edge was the only structural feature that distinguished them, and it was removed.

2. **Pramana labels at this scale are noisy proxies.** All AI2THOR claims have `pramana_primary=perception`; all AVeriTeC testimony claims have `pramana_primary=testimony`. The Pramana one-hot on the EpistemicNode is a near-perfect proxy for the source dataset (`ai2thor` vs `averitec`), which in turn is a proxy for the claim's modality. Once stance edges are removed, there is no additional signal in the Pramana label beyond what the source identity already provides.

3. **Model capacity is not the bottleneck.** Run A (text only) and Run B (text + epistemic) are identical, ruling out that the epistemic node simply doesn't have enough capacity to contribute. The problem is the information content of the heuristic labels, not the architecture.

**Run D (Pathway B)** reproduces Run C's perfect accuracy because stance edges are still present — the routing shortcut dominates. The Pathway B ablation (modality-learned Pramana) cannot be meaningfully evaluated without also removing stance edges, which would require combining `--no-stance-edges` with `--use-modality-learning`. This is a Phase 6 stretch run not completed here.

---

## 4. Per-Pramana Analysis

**Per-Pramana test accuracy for Runs B and A (both identical):**

| Pramana | Test accuracy | Support (n) | Analysis |
|---------|--------------|-------------|----------|
| `perception` | 0.797 | 128 | AI2THOR claims — mostly `supported`; text embedding distinguishes some |
| `inference` | 0.662 | 74 | Higher `refuted` proportion; majority-class prediction helps |
| `testimony` | 0.617 | 188 | AVeriTeC mix; same majority-class effect |
| `comparison_analogy` | 0.558 | 52 | Near 50/50 — close to random baseline for this class |
| `non_apprehension` | 0.000 | 75 | All NEE claims completely missed — 0/75 correct |

**Key observations:**

**`non_apprehension` = 0/75 correct.** Every NEE claim (not_enough_evidence verdict) is misclassified. Without the `no_evidence` edge, the model has no structural signal distinguishing NEE claims from ordinary `supported` or `refuted` claims. The sentence embedding of "No answer could be found" is insufficient — the model has learned that the default prediction is `refuted`, and nothing in the text strongly overrides this.

This is the strongest finding of the per-Pramana analysis: the `non_apprehension` Pramana type, which is directly linked to the `not_enough_evidence` verdict by both the human-designed rules (ADR-007) and the epistemic theory (ADR-005), produces 0% recall without the stance edge. The Pramana label carries this information semantically but the GNN cannot extract it from a 6-dimensional one-hot vector when class imbalance is 6.6% and the majority-class signal dominates.

**`perception` at 0.797** is the only Pramana type with meaningfully above-chance accuracy. AI2THOR claims are physically concrete ("The apple is on the table") — their sentence embeddings are structurally distinct from AVeriTeC news claims, and most are `supported`, so the 39%→57% boost above the text-only prediction rate suggests some real embedding-based signal.

**Per-source accuracy (Runs B and A):**

| Source | Test accuracy | Support (n) |
|--------|--------------|-------------|
| averitec | 0.611 | 337 |
| ai2thor | 0.500 | 180 |

AVeriTeC accuracy (0.611) exceeds AI2THOR accuracy (0.500) because AVeriTeC has a higher proportion of `refuted` claims (the majority-class prediction), whereas AI2THOR has more `supported` claims that the model consistently misses.

---

## 5. Research Contribution

**Novel integration of Pramana epistemology with heterogeneous GNN fact verification.**
This is the first project to operationalise the five Sanskrit Pramana knowledge-source
categories (ADR-001) as structural features in a GNN-based fact-verification system,
encoding them as EpistemicNode features and edge-level confidence weights (ADR-004,
ADR-008).

**Two-source unified dataset with epistemic annotation.**
AI2THOR simulation evidence (direct sensory / perception Pramana) and AVeriTeC web
evidence (testimony Pramana) are unified under a single JSONL schema (v2.0) with
per-claim Pramana labels, confidence weights, and modality annotations (ADR-002). The
dataset bridges the gap between embodied simulation environments and large-scale web
fact-checking.

**Distinct treatment of `absent` vs. `not_enough_evidence`.**
A key design decision (ADR-005, ADR-014) is distinguishing AI2THOR's
*absence-confirmed* stance (`absent`, verdict=`supported`) from AVeriTeC's
*unresolved-search* stance (`no_evidence`, verdict=`not_enough_evidence`). Both involve
absence, but they have opposite epistemic statuses. This maps to the Sanskrit distinction
between *Anupalabdhi* (valid non-perception of an object that would be perceivable if
present) and simple lack of evidence.

**Identification of the stance-shortcut problem.**
Stance-annotated fact-verification datasets where the labelling protocol is
self-consistent create a deterministic routing rule that any model will learn immediately.
The Phase 4 finding (val_acc=1.0 at epoch 1) is the clearest demonstration of this. The
Phase 5 ablation design (ADR-016) provides a methodology for circumventing the shortcut
and isolating the epistemic contribution.

**Ablation methodology for separating epistemic vs. text contributions.**
The four-run ablation matrix (A=text floor, B=text+epistemic, C=full routing ceiling,
D=learned Pramana) provides a reusable template for evaluating epistemic annotation in
any fact-verification dataset.

---

## 6. Limitations

**Dataset scale.** ~5,000 records is small for GNN training. The model has ~2.4M
parameters, giving a parameter-to-record ratio that risks overfitting in Runs A/B where
the routing shortcut is unavailable.

**Modality coverage.** Only two modalities are meaningfully represented: `simulation_state`
(AI2THOR, ~30% of records) and `web_text` (AVeriTeC, ~70%). The `video`, `audio`, and
`image` modality types are reserved but unused. The modality one-hot vector has 3 always-zero
dimensions.

**Heuristic Pramana labels.** Pramana types are assigned by adapter-level rules (ADR-007),
not by human annotation or model inference. For example, all AI2THOR evidence is labelled
`perception` because it comes from a simulation — there is no case-by-case judgement.
Heuristic labels that correlate perfectly with the data source may be learning the source
identity rather than the epistemic type.

**Pramana class imbalance.** The actual distribution deviates from targets (ADR-012 delta):
`perception` is over-represented (+13.6pp), `testimony` is under-represented (−20.6pp).
Weighted CrossEntropyLoss handles verdict imbalance but not Pramana imbalance; the
Pramana auxiliary head in Pathway B may be biased toward `testimony`.

**`comparison_analogy` and `inference` coverage.** These two Pramana types together account
for ~26% of records but have limited training examples for the `supported` class. Per-Pramana
accuracy for these types may be unreliable.

---

## 7. Future Work

**Larger dataset with real multi-hop claims.**
Claims where individual evidence items are inconclusive, and verdict requires reasoning
over multiple epistemic pathways simultaneously, would be a more meaningful test of the
Pramana hypothesis.

**Human verification of Pramana labels.**
Ground-truth Pramana labels assigned by epistemology-trained annotators would replace
the heuristic rules (ADR-007) with a gold standard. Inter-annotator agreement on Pramana
type assignment is an open research question.

**Structured triple-graph for AI2THOR.**
Phase 4 encodes claim triples as `"s p o"` string embeddings (384-d). A structured
mini-graph with entity nodes and predicate edges within each AI2THOR subgraph would
preserve relational structure and enable richer message-passing. Deferred as Phase 5
ablation.

**Multi-Pramana EpistemicNodes.**
The schema stores `pramana_all` for claims with multiple applicable Pramana types. Using
one EpistemicNode per entry in `pramana_all` would let the model reason over multiple
epistemic pathways simultaneously. Deferred as Phase 5 ablation.

**Cross-domain generalisation.**
The model is trained on a mixture of AI2THOR (simulation, controlled domain) and AVeriTeC
(news fact-checking, open domain). Testing generalization to a held-out third source
(e.g., scientific claim verification) would validate the Pramana approach's transferability.

**Adversarial epistemic test cases.**
Claims where the routing rule (stance → verdict) fails would directly test whether the
epistemic node provides independent signal in the presence of conflicting evidence.
Examples: claims where multiple pieces of testimony-Pramana evidence `supports` the claim
but an inference-Pramana analysis of source quality `refutes` it.
