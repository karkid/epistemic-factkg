# ADR-001: Pramana Epistemic Framework

## Status

Accepted

## Context

Fact verification systems typically treat all evidence as epistemically equivalent — a sensor reading, a web article, and an inference from multiple sources are all weighted the same during classification. This makes sense for pure NLP benchmarks but produces opaque confidence estimates in a GNN training context, where edge weights should reflect the *reliability* of the knowledge source, not just whether the evidence is present.

We needed a principled way to assign epistemic confidence priors to claims from two fundamentally different sources:
- **AI2-THOR**: synthetic simulation with perfect ground truth, perception-based
- **AVeriTeC**: real-world web text, testimony and inference-based

Alternatives considered:
- **Uniform weights** — simple, but provides no epistemic signal; every evidence type gets the same prior
- **Purely statistical priors** — learned from data, but we have no labelled calibration set at this stage; also circular if trained on the same claims
- **Domain-expert annotation** — expensive and slow; impractical for dataset construction at scale

## Decision

Adopt the six-Pramana taxonomy from Sanatana Dharma as the epistemic label set, with heuristically assigned confidence weights. The Pramana system classifies knowledge sources into six categories that map naturally onto the evidence types present in our data.

See [ADR-007](007-heuristic-epistemic-labeling.md) for how labels are assigned and [ADR-008](008-heuristic-prior-weight-values.md) for how the weight values were chosen.

The Pramana categories and their mappings:

| Pramana | Label | Primary use |
|---|---|---|
| Pratyakṣa | `perception` | AI2-THOR simulation state — sensor-confirmed facts |
| Anupalabdhi | `non_apprehension` | Absence claims — sensor-confirmed missing object/state |
| Upamāna | `comparison_analogy` | Numeric or analogy-based claims |
| Śabda | `testimony` | AVeriTeC — single-source web text |
| Anumāna | `inference` | AVeriTeC — multi-source or synthesised reasoning |
| Arthāpatti | `postulation_derivation` | Derived/assumed facts; limited use, lowest reliability |

## Consequences

**Positive:**
- Confidence priors are theoretically grounded, not arbitrary hyperparameters
- Extends naturally to new data sources — any new dataset's `infer_pramana()` maps onto existing categories
- GNN edge weights encode epistemic reliability, enabling the model to learn whether high-Pramana claims are more predictive

**Negative:**
- Requires explaining the Pramana framework to new contributors unfamiliar with Indian epistemology
- Every new adapter must implement `infer_pramana()` — a non-trivial requirement
- The six categories do not cover every possible evidence type; future edge cases may require extending or subdividing
- Weights are theoretical priors that have not yet been empirically validated — Phase 5 ablation will test whether they improve model performance
