# ADR-007: Reduce Verdict Classes from 4 to 3 — Drop conflicting_evidence

## Status

Accepted

## Context

The unified schema defines 4 verdict labels: `supported`, `refuted`, `not_enough_evidence`, `conflicting_evidence`. The Phase 4 training dataset distribution after ADR-005 filtering is:

| Verdict | Count | % |
|---|---|---|
| refuted | 2,947 | 54.9 |
| supported | 1,871 | 34.9 |
| not_enough_evidence | 317 | 5.9 |
| conflicting_evidence | 233 | 4.3 |

Options considered for the 4→3 or 4→2 class reduction:

| Option | Description | Problem |
|---|---|---|
| **Keep all 4 classes** | Train on all 4 verdict labels | `conflicting_evidence` at 4.3% (233 records) is insufficient for reliable GNN learning |
| **Drop to binary** | Keep only supported/refuted | Loses `not_enough_evidence` — this class is the direct output of `non_apprehension` reasoning (ADR-001), central to the research contribution |
| **Drop conflicting_evidence only** | 3-class: supported, refuted, not_enough_evidence | Removes weakest, least epistemically grounded class; preserves the full epistemic story |
| **Rebuild with more supported claims** | Regenerate AI2THOR claims to balance classes | `claims_all.jsonl` is frozen (ADR build design); requires running the simulator; disproportionate effort for a distribution problem solvable by loss weighting |

## Decision

**Use 3-class verdict classification. Exclude `conflicting_evidence` records from GNN training.**

Verdict label mapping:
- 0 = `supported`
- 1 = `refuted`
- 2 = `not_enough_evidence`

Post-exclusion distribution:
| Verdict | Count | % |
|---|---|---|
| refuted | 2,947 | 61.4 |
| supported | 1,871 | 39.0 |
| not_enough_evidence | 317 | 6.6 |
| **Total** | **5,135** | |

Remaining class imbalance handled with **weighted `CrossEntropyLoss`** — inverse-frequency weights computed via `EpistemicFactDataset.get_class_weights()`.

**Implementation:** Extend `filter_for_training.py` to also exclude records with `verdict.label == "conflicting_evidence"`. The `Verdict.CONFLICTING_EVIDENCE` enum value remains defined in `src/core/claims/labels.py` for schema completeness.

## Consequences

**Why `conflicting_evidence` is dropped:**
- 233 records (4.3%) is insufficient for reliable class representation learning in a GNN
- The label is an AVeriTeC-specific artifact: it arises when two QA evidence pairs contradict each other, not from an epistemically distinct reasoning type
- No Pramana type maps cleanly to conflicting evidence — it is primarily a data quality signal, not an epistemic category
- Removing it does not weaken the epistemic framework; the 3 remaining classes each map to distinct Pramana patterns

**Why `not_enough_evidence` is kept:**
- 317 records (5.9%) is small but above the threshold for meaningful class learning with loss weighting
- This class is the direct verdict output of `non_apprehension` (Anupalabdhi) reasoning — it represents claims where absence of evidence in a closed world is itself a knowledge assertion (ADR-001)
- Dropping it would eliminate the most epistemically interesting verdict class and undermine the research contribution

**Why no rebuild:**
- `claims_all.jsonl` is intentionally frozen to decouple the pipeline from the AI2THOR simulator
- The class imbalance after dropping `conflicting_evidence` (61.4% / 39.0% / 6.6%) is manageable with inverse-frequency loss weighting, a standard and well-justified approach
- Rebuilding introduces randomness and reproducibility risk; loss weighting is deterministic and well-understood

**Future work:**
`conflicting_evidence` records are retained in the unified JSONL and filtered only at the GNN training stage. If a future dataset provides sufficient conflicting-evidence claims (>500), the filter can be removed and the model extended to 4-class output without schema changes.
