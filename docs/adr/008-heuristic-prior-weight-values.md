# ADR-008: Heuristic Prior Weight Values

## Status

Accepted

## Context

Having chosen the six Pramana categories (ADR-001) and heuristic labeling rules (ADR-007), we needed to assign a numeric confidence weight to each Pramana. These weights serve as edge priors in the GNN graph — they encode the *expected epistemic reliability* of each evidence type before any data-driven learning.

Options for determining weights:

| Approach | Description | Problem |
|---|---|---|
| **Uniform** | All Pramana = 0.5 | Discards all epistemic information; defeats the purpose |
| **Empirically estimated** | Learn weights from a calibration dataset | No calibration ground truth at this stage; would require a separate annotation effort |
| **Theoretically motivated** | Assign weights based on known properties of each knowledge source | Requires domain reasoning; not backed by data yet |

## Decision

Use **theoretically motivated weights** based on the epistemic reliability of each knowledge source type. Values are defined in `src/core/claims/labels.py:CONFIDENCE_WEIGHTS`.

| Pramana | Weight | Rationale |
|---|---|---|
| `perception` | **0.95** | AI2-THOR simulator is a closed-world environment — ground truth is exact. Near-certain but not 1.0 (simulator may not model all real-world physics) |
| `testimony` | **0.80** | Web text sources are generally reliable but carry noise, bias, and potential misinterpretation |
| `non_apprehension` | **0.75** | Absence of evidence is a valid epistemic act in closed-world settings (AI2-THOR), but weaker than positive testimony in open-world settings (AVeriTeC) |
| `comparison_analogy` | **0.65** | Analogical or statistical reasoning introduces uncertainty from the analogy itself; benchmarks may not apply precisely |
| `inference` | **0.55** | Multi-hop or multi-source reasoning compounds error at each step; the more steps, the more opportunity for error propagation |
| `postulation_derivation` | **0.40** | Hypothetical derivation — assumes a fact to explain a circumstance. Least reliable; used sparingly |

### Key properties of the weight scale

- No weight is 0.0 or 1.0 — even the best source (perception) is not infallible, and even the weakest (postulation) contributes something
- Weights are conservative mid-range values, not extreme — this reflects uncertainty about the true reliability of each category
- The ordering (perception > testimony > non_apprehension > comparison > inference > postulation) reflects classical Pramana epistemological ranking

## Consequences

**Positive:**
- Weights are theoretically justified and can be cited in the research paper
- The ordering provides meaningful signal even before GNN training
- Conservative non-extreme values reduce the risk of catastrophic over- or under-confidence

**Negative:**
- Weights are not empirically validated — they represent theoretical priors that may not match the actual verification accuracy distribution in the data
- Phase 5 ablation studies will test whether these priors improve or hurt GNN performance vs. uniform weights; if they hurt, they should be revised
- The values may need to be updated if future data reveals systematic miscalibration (e.g., if `inference` claims turn out to be highly accurate in practice)

**Revision policy:**
Update `CONFIDENCE_WEIGHTS` in `src/core/claims/labels.py` and update this ADR if empirical results from Phase 5 suggest different values. Do not update weights mid-experiment.
