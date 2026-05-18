# Labeling Guide — Epistemic FactKG v3.0

This guide covers how each per-evidence field is assigned during dataset construction. It is the authoritative reference for adapter developers and for understanding the epistemic model.

The four fields that must be set on every `evidence[]` item are:

| Field | Set by | How |
|---|---|---|
| `evidence_types` | Converter | Modality + content heuristics (below) |
| `source_id` | Converter | URL domain lookup → registry key |
| `inference_strength` | Converter | Answer-type rubric (below) |
| `confidence_weight` | Converter | EC formula: $1 - (1-ST)^{EW \times IS}$ |

`stance` is set by the source adapter (AI2THOR: from simulation result; AVeriTeC: from answer content; Synthetic: from template spec).

---

## Section 1 — `evidence_types` assignment

`evidence_types` is a per-evidence multi-label list. It replaces the old claim-level `pramana_primary`.

### AI2THOR (automated, no ambiguity)

| Claim structure | `evidence_types` |
|---|---|
| `one_hop`, `conjunction`, `negation` | `["perception"]` |
| `absence` | `["non_apprehension"]` |
| Spatial relation (near/far/distance) | `["perception", "comparison_analogy"]` |
| Action-based (tried and verified result) | `["perception", "inference"]` |

These are set deterministically in `AI2ThorConverter` from `claim_type`.

### AVeriTeC (heuristic — modality-first, content overrides on top)

**Step 1 — Start from modality:**

| `modality` | Base `evidence_types` |
|---|---|
| `image` / `video` / `audio` | `["perception"]` |
| `web_text` / `pdf` | `["testimony"]` |
| `web_table` | `["comparison_analogy", "testimony"]` |
| `annotation_knowledge` | `["testimony"]` |
| `unanswerable` | `[]` → stance forced to `not_enough_evidence` |
| `other` | `["testimony"]` (conservative fallback) |

**Step 2 — Apply content overrides (append, do not replace):**

- Numeric/statistical cues in answer text (`%`, `GDP`, `rank`, `million`, `billion`, ordinal numbers) → append `"comparison_analogy"` if not already present
- Abstractive answer + ≥2 distinct source URLs → append `"inference"` if not already present

**Step 3 — URL pattern signals** (feed into `source_id` resolution, not `evidence_types` directly):

| URL pattern | `source_type` |
|---|---|
| `.gov`, `.gov.uk`, `.gov.in`, `.gc.ca` | `government` |
| `.edu` | `scientific_paper` |
| `wikipedia.org` | `knowledge_graph` |
| `twitter.com`, `x.com`, `facebook.com`, `reddit.com`, `instagram.com` | `social_media` |
| `.org` (unrecognised) | `web_text` (default trust tier) |

### Synthetic (set by template spec, not heuristic)

Templates fix `evidence_types` at definition time. The text client does not influence `evidence_types`.

---

## Section 2 — `stance` assignment

**Key principle: stance is a content label, not a reliability label.**

A low-trust source that clearly supports the claim → `stance = supports`, but `confidence_weight` ($EC_i$) will be low because $ST$ is low. Do not use `not_enough_evidence` stance as a proxy for low trust. Source reliability is handled entirely by $ST_i$ in the formula.

**Decision tree (apply in order):**

1. Sensor-confirmed absence in a closed-world environment? → `absent`
   *(AI2THOR `non_apprehension` only; not applicable to text sources)*

2. Evidence directly and unambiguously confirms the claim, no hedging? → `supports`

3. Evidence directly and unambiguously contradicts the claim, no hedging? → `refutes`

4. A single evidence item internally contradicts itself (asserts both X and not-X)? → `conflicting_evidence`

5. Evidence is hedged, partial, or ambiguous? → `not_enough_evidence`
   - Hedging triggers: *"reportedly"*, *"allegedly"*, *"may indicate"*, *"possibly"*, *"sources suggest"*, *"is believed to"*, *"could suggest"*
   - Partial: answer covers only part of the claim's assertion
   - Unanswerable: AVeriTeC `answer_type = unanswerable`

**$EC_i$ floor override:** If the computed `confidence_weight` falls below `MIN_EVIDENCE_CONFIDENCE = 0.10`, override stance to `not_enough_evidence` regardless of content. Evidence below this floor contributes negligibly to aggregation and should not be treated as meaningful support or refutation.

**Note on multi-evidence conflicts:** When two *separate* evidence items contradict each other, each gets its own content stance (`supports` and `refutes` respectively). The verdict becomes `conflicting_evidence` through score aggregation — not through individual item stances. `conflicting_evidence` at the evidence level applies *only* to a single item that internally contradicts itself.

---

## Section 3 — `inference_strength` (IS) rubric

IS measures how many inferential steps separate the evidence from the claim conclusion. It is independent of source quality (that is $ST$).

| IS | Description | Typical cases |
|---|---|---|
| 1.0 | Direct ground truth, no inference required | Simulator state; primary sensor measurement |
| 0.8 | One step, extractive | Direct quote; official record; Boolean QA; extractive answer |
| 0.6 | Abstractive or multi-source synthesis | Abstractive QA answer; secondary source; one synthesis step |
| 0.4 | Partial or circumstantial | Evidence covers only part of the claim; related but indirect |
| 0.2 | Speculative or highly indirect | Expert opinion without stated basis; prediction; extrapolation |
| 0.0 | Unanswerable | AVeriTeC `answer_type = unanswerable`; no evidence content |

**AVeriTeC `answer_type` heuristics:**

| `answer_type` | IS |
|---|---|
| `Boolean`, `Extractive` | 0.8 |
| `Abstractive` (single source URL) | 0.6 |
| `Abstractive` (≥2 distinct source URLs) | 0.6 |
| `Unanswerable` | 0.0 |

**AI2THOR:** always 1.0 (simulation ground truth, direct observation).

**Synthetic:** IS is fixed by template spec (e.g., `inference_nee` template sets IS=0.5 to produce NEE verdict).

---

## Section 4 — `source_id` and source trust assignment

`source_id` is the registry key that resolves to `source_trust` ($ST$) via `data/registry/source_trust_registry.jsonl`.

**Format:** `{domain}_{modality}` — e.g., `bbc_web_text`, `wikipedia_knowledge_graph`, `sensor_perception`.

**Fallback:** unresolvable `source_id` → `DEFAULT_SOURCE_TRUST = 0.30`.

**Source-type default tiers** (applied when no domain-specific entry exists):

| Source type | Default $ST$ | Basis |
|---|---|---|
| `sensor` (AI2THOR / IoT) | 0.95 | Direct perception / closed-world measurement |
| `scientific_paper` | 0.90 | Peer-review process |
| `government` | 0.85 | Official records |
| `knowledge_graph` (Wikipedia) | 0.80 | Community-reviewed, cited |
| `news_media` (high credibility) | 0.75 | MBFC "High" factual rating |
| `news_media` (mixed/general) | 0.65 | MBFC "Mostly Factual" |
| `web_text` (general) | 0.55 | Unverified web source |
| `llm_generated` | 0.50 | Synthetic, no real-world grounding |
| `social_media` | 0.35 | Self-published, unverified |
| `unknown` | 0.30 | Fallback — unknown source ranks below known-bad |

Domain-specific overrides use MBFC ratings. Each override is documented with `trust_metadata.methodology_ref = "MBFC:{slug}"` in the registry record.

See [ADR-015](adr/015-source-trust-registry.md) for the full registry design and Bayesian recalibration path.

---

## Section 5 — `confidence_weight` (EC) formula

Per-evidence confidence is computed and stored at convert time:

$$EC_i = 1 - (1 - ST_i)^{EW_i \times IS_i}$$

- $ST_i$ — source trust, from registry via `source_id`
- $EW_i$ — `combine_pramana_weights(evidence_types)` — diminishing-returns combination of type weights
- $IS_i$ — inference strength (Section 3)

**Type weights** (from `CONFIDENCE_WEIGHTS` in `labels.py`):

| Evidence type | Weight |
|---|---|
| `perception` | 0.95 |
| `non_apprehension` | 0.85 |
| `testimony` | 0.75 |
| `inference` | 0.70 |
| `comparison_analogy` | 0.65 |

For multi-type evidence items, `combine_pramana_weights` applies diminishing returns across the list.

---

## Section 6 — Verdict derivation

Verdict is derived from per-evidence confidence scores via product-of-complements aggregation:

$$\text{SupportScore} = 1 - \prod_{i \in \text{Supports} \cup \text{Absent*}}(1 - EC_i)$$
$$\text{RefuteScore} = 1 - \prod_{i \in \text{Refutes}}(1 - EC_i)$$

*`absent` stance counts toward SupportScore for non_apprehension claims (confirmed absence supports the claim that the object is absent).

Evidence with `not_enough_evidence` or `conflicting_evidence` stance is excluded from both aggregations.

**Verdict thresholds:**

```
support_score >= 0.75 AND refute_score < 0.40   → supported
refute_score  >= 0.75 AND support_score < 0.40  → refuted
support_score >= 0.40 AND refute_score >= 0.40  → conflicting_evidence
else                                             → not_enough_evidence
```

Threshold rationale: 0.75 = high-confidence belief threshold; 0.40 = non-trivial opposition threshold. The strict `< 0.40` guards mean that a RefuteScore exactly at 0.40 blocks a "supported" verdict.

For AI2THOR and AVeriTeC records, `derivation_method = "annotated"` — the original dataset verdict is kept, not re-derived. For synthetic records, `derivation_method = "aggregated_from_evidence"` — the template math determines the verdict by construction.

See [ADR-019](adr/019-per-evidence-epistemic-modeling.md) for the full formula derivation and threshold justification.
