# ADR-019: Inference Strength Rubric for AVeriTeC Evidence

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-010 (per-evidence IS), ADR-009 (source trust registry)

---

## Context

AVeriTeC provides an `answer_type` field per evidence item with values:
`boolean`, `extractive`, `abstractive`, `unanswerable`.

Unlike AI2THOR (triple-based, IS derivable from relation type) or synthetic
(IS specified in template), AVeriTeC has no explicit IS field. A principled
IS assignment rule is needed that (a) reflects epistemic directness and
(b) is consistent with the IS scale used in other sources.

---

## Decision

Assign IS from `answer_type` using the following rubric:

| answer_type  | IS  | Rationale |
|--------------|-----|-----------|
| boolean      | 0.8 | Direct yes/no lookup — high inferential directness |
| extractive   | 0.8 | Verbatim span from source — high directness |
| abstractive  | 0.6 | Synthesised from source — some inferential gap |
| unanswerable | 0.0 | No usable evidence — no inference possible |

This IS is then further bounded by source trust (see ADR-021): a low-trust
source cannot contribute IS above its trust score regardless of answer_type.

---

## Alternatives Considered

**A. Uniform IS = 0.6 for all AVeriTeC evidence** — ignores the meaningful
signal in answer_type. Rejected as unnecessarily lossy.

**B. IS from fact-checking strategy field** — strategies like `consultation` vs
`written evidence` map onto epistemic directness. Partially implemented as
evidence_type enrichment but not used for primary IS assignment because
`answer_type` is more directly tied to the evidence item's inferential role.

**C. Learned IS for AVeriTeC (no supervision)** — ISHead would learn IS
from verdict supervision alone. Rejected: the IS must be interpretable and
grounded; unsupervised IS assignment defeats the epistemic purpose.

---

## Consequences

- IS targets for AVeriTeC are heuristic, not annotated — this is acknowledged
  as a data limitation in the paper.
- Boolean/extractive dominate AVeriTeC (IS = 0.8); model IS predictions cluster
  near 0.8 for these records.
- The IS cap (ADR-021) adjusts these values downward for low-trust sources,
  making the final IS distribution more realistic.
