# ADR-021: AVeriTeC Evidence-Type Labeling Rules

**Status:** Accepted  
**Date:** 2026-05-15  
**Supersedes:** Portions of ADR-007 (AVeriTeC-specific claim-level pramana rules)  
**Related:** ADR-019 (per-evidence modeling), ADR-018 (source trust registry)

---

## Context

AVeriTeC is a web-sourced fact-checking dataset. Each claim has:
- `fact_checking_strategies` (claim-level list) — how annotators verified the claim
- Per-answer `source_medium` — the type of the cited source
- Per-answer `answer_type` — boolean, extractive, abstractive, unanswerable

ADR-007's `_infer_pramana` assigned a single `pramana_primary` per claim by aggregating across all evidence items. Schema v3.0 requires per-evidence `evidence_types`, which must be assigned individually.

---

## Decision

**Use a three-pass assignment for each evidence item's `evidence_types`, then apply a claim-level strategy enrichment pass.**

### Pass 1 — Modality-based base types (per item)

`source_medium` is mapped to a modality, which drives the base `evidence_types`:

| Modality | `evidence_types` | Rationale |
|---|---|---|
| `image`, `video`, `audio` | `["perception"]` | Direct sensory observation; no inference needed |
| `web_table` | `["comparison_analogy", "testimony"]` | Tabular data is inherently comparative; also cited as written evidence |
| `web_text`, `pdf`, `other` | `["testimony"]` | Written/cited source — the default for AVeriTeC |
| `annotator_knowledge` | `["testimony"]` | Annotator's own knowledge counts as testimony |
| `unanswerable` | `[]` | No real evidence; stance forced to `not_enough_evidence` |

**Why perception has no inference**: Perceptual evidence is direct observation — the viewer/annotator observes the thing. No reasoning step is required to establish *what* the evidence is. If the annotator then synthesises across perceptual items, that synthesis is captured by the multi-source abstractive pass (Pass 2), not applied blindly to all perceptual items.

### Pass 2 — Numeric cue detection (per item, textual only)

For textual items (web_text, pdf), if the answer text contains numerical/statistical comparison cues (`%`, GDP, million, rank, largest, etc.) → append `comparison_analogy`.

This captures claims verified by comparing magnitudes or statistical benchmarks where the web_text content itself is the signal, independent of the claim-level strategy.

### Pass 3 — Multi-source abstractive inference (per item)

If **total distinct source URLs across the claim ≥ 2** AND the item's `answer_type == abstractive` → append `inference`.

Rationale: an abstractive answer synthesising evidence from multiple sources constitutes multi-step reasoning. A single-source abstractive answer is a summary, not multi-hop inference.

### Pass 4 — `fact_checking_strategies` enrichment (per item, textual only)

Claim-level `fact_checking_strategies` carries information about the verification method that is not always recoverable from individual answer texts. The enrichment rules:

| Strategy | Evidence type added | Condition |
|---|---|---|
| `Numerical Comparison` | `comparison_analogy` | Item is textual (not perceptual), not unanswerable |
| `Consultation` | `inference` | Item is textual, not unanswerable |

**Perceptual items are excluded from enrichment.** Strategy labels describe what the *annotator* did with the evidence, not the nature of the sensory observation itself. Applying "Numerical Comparison" to a video clip would be incorrect — the video is still direct perception; the comparison happens after the observation.

`Written Evidence` is not mapped — testimony is already assigned by Pass 1.

---

## `inference_strength` heuristics

Inference strength (`IS_i`) measures the directness of the evidence-to-claim link, following the rubric in ADR-019:

| `answer_type` | IS | Rationale |
|---|---|---|
| `boolean` | 0.8 | Direct yes/no lookup, one step |
| `extractive` | 0.8 | Direct quote or value extraction, one step |
| `abstractive` | 0.6 | Synthesis or paraphrase, potentially multi-step |
| `unanswerable` | 0.0 | No evidence found; no link to claim |

---

## `source_id` resolution

Each evidence item's `source_id` is resolved from `source_url` via the source trust registry (ADR-018):
1. Parse domain from URL; strip `www.`, strip TLD (e.g. `reuters.com` → `reuters`)
2. Exact registry match: `{name}_{modality}`
3. Subdomain strip: try parent domain
4. TLD heuristic: `.gov` → `government_web_text`, `.edu` → `general_web_text`
5. Social media check
6. Modality default (`web_table`, `pdf`)
7. Fallback: `unknown_web` (ST = 0.40)

---

## Consequences

- Per-evidence types are richer than the old claim-level `pramana_primary`: a single claim can have `["testimony"]` items alongside `["testimony", "comparison_analogy"]` items.
- `evidence_types_all` in the epistemic block is the union across all items — a summary for filtering and GNN readiness checks.
- `non_apprehension` is never assigned to AVeriTeC records. Web text cannot confirm closed-world absence; this type is restricted to AI2THOR (ADR-005).
- The strategy enrichment pass is applied after the multi-source inference pass, so a "Consultation" strategy does not accidentally remove inference that was already added by Pass 3.
