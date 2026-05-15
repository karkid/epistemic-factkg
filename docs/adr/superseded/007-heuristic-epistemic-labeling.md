# ADR-007: Heuristic Rule-Based Epistemic Labeling

## Status

**SUPERSEDED by [ADR-019](019-per-evidence-epistemic-modeling.md)**

Reason: ADR-007 assigned a single `pramana_primary` label per claim via heuristic rules. ADR-019 abolished `pramana_primary` entirely, moving to per-evidence multi-label `evidence_types` with a formal confidence formula. The heuristic rules described here are preserved in `docs/adr/superseded/` for methodology history.

## Context

Every record in the unified dataset requires a `pramana_primary` label and a `confidence_weight`. Assigning these labels requires choosing a strategy:

| Strategy | Description | Problem |
|---|---|---|
| **Manual annotation** | Human experts assign Pramana labels per record | Expensive, slow, inconsistent across annotators at scale |
| **Crowdsourcing** | Non-expert annotators label records | Hard to enforce schema consistency; Pramana categories require domain knowledge |
| **ML-based classification** | Train a model to predict Pramana labels | Circular — we're training on the same claims we're labeling; no ground truth |
| **Rule-based heuristics** | Deterministic rules per data source | Fast, consistent, reproducible, auditable |

## Decision

Use **deterministic rule-based heuristics** implemented per adapter in `infer_pramana()`.

The rules are based on observable structural properties of each source record — not the claim text content — to ensure determinism and avoid introducing NLP noise at the labeling stage.

### AI2-THOR labeling rules

| Condition | `pramana_primary` | Rationale |
|---|---|---|
| Evidence from sensor/simulation state | `perception` | Direct observation in closed-world simulator |
| Absence claim (`evidence[].stance = "absent"`) | `non_apprehension` | Sensor-confirmed absence (Anupalabdhi) |
| Numeric/spatial comparison | `comparison_analogy` | Analogical or comparative reasoning |

### AVeriTeC labeling rules

| Condition | `pramana_primary` | Rationale |
|---|---|---|
| Single evidence source | `testimony` | Direct web document — Śabda |
| Multiple sources or synthesised answer | `inference` | Multi-hop reasoning — Anumāna |
| Verdict = `not_enough_evidence` | `non_apprehension` | Absence of sufficient textual knowledge — see [ADR-005](005-anupalabdhi-distinct-from-not-enough-evidence.md) |

## Consequences

**Positive:**
- Consistent and reproducible — same input always produces the same label
- Fast — no external calls or model inference required during dataset construction
- Auditable — the rules are visible in each adapter's `infer_pramana()` method
- Extensible — new datasets implement their own rules without changing existing ones

**Negative:**
- Labels are heuristic priors, not ground truth — they represent our best guess based on source structure, not verified epistemic categories
- Edge cases (a claim that is both perception-based and inference-based) may be under-annotated; only one `pramana_primary` is assigned
- The labeling rules may be wrong for some records — Phase 5 GNN ablation will validate whether the labels carry signal or introduce noise

**Note on `pramana_all`:**
Records can list multiple applicable Pramana labels in `pramana_all` even when `pramana_primary` is a single dominant label. Multi-label assignment uses the diminishing returns formula from [ADR-006](006-diminishing-returns-combination-formula.md).
