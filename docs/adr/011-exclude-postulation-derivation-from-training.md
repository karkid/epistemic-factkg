# ADR-011: Exclude postulation_derivation from GNN Training

## Status

Accepted

## Context

The six Pramana categories are all defined in `src/core/claims/labels.py` and the unified schema. However, `postulation_derivation` (Arthāpatti) is the rarest category in practice — it covers claims verified by assuming a necessary fact to explain a circumstance, which is the most hypothetical and least reliable reasoning type (confidence weight: 0.40).

Across the full dataset (~6,250 claims), `postulation_derivation` accounts for approximately 50 records — exclusively from AVeriTeC edge cases. This is insufficient for a GNN to learn a meaningful node representation for that Pramana type. Training with ~50 samples in a class typically produces an under-fitted, noise-dominated representation that can destabilise the overall model.

Options considered:

| Option | Description | Problem |
|---|---|---|
| **Keep in training** | Train on all 6 Pramana types including postulation | ~50 samples → near-empty class; GNN learns noise, not signal |
| **Merge with inference** | Treat `postulation_derivation` as `inference` during training | Epistemically similar (both low-confidence derived reasoning), but masks a real distinction |
| **Exclude from training** | Filter out postulation records; keep the category defined for future use | Clean training distribution; deferred, not deleted |

## Decision

**Exclude `postulation_derivation` claims from GNN training.** The category remains fully defined in the schema, code, and ADRs — it is deferred to future work, not removed from the framework.

Concretely:
- Records with `epistemic.pramana_primary = "postulation_derivation"` are filtered out before graph construction in Phase 4
- The GNN trains on 5 Pramana types: `perception`, `testimony`, `non_apprehension`, `comparison_analogy`, `inference`
- `CONFIDENCE_WEIGHTS` in `src/core/claims/labels.py` retains the 0.40 entry — it is used by `combine_pramana_weights()` if postulation appears in `pramana_all` of a multi-Pramana record
- The filter is applied at the dataset loading stage, not by modifying source JSONL files

## Consequences

**Positive:**
- Training distribution is clean — no near-empty class degrading GNN representations
- The 5 remaining Pramana types have sufficient samples for meaningful representation learning
- The framework is honest: the category is excluded with a documented reason, not silently dropped

**Negative:**
- Claims involving postulation-style reasoning from AVeriTeC are excluded entirely — a small information loss
- If a future dataset adds postulation-heavy content, the exclusion logic must be revisited
- Research paper must clearly note the 5-class Pramana training setup (not 6) to avoid misleading readers about the framework's scope

**Future work:**
Revisit when a dataset with sufficient postulation-style claims (>500) becomes available. Arthāpatti is epistemically distinct and worth modelling eventually — it represents the weakest but still valid form of knowledge, which is relevant to borderline `not_enough_evidence` cases.
