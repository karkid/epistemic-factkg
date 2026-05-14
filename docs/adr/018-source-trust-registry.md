# ADR-018: Source Trust Registry

## Status

Accepted

## Context

The v3.0 confidence formula requires a source trustworthiness score $ST_i$ for each evidence item:

$$EC_i = 1 - (1 - ST_i)^{EW_i \times IS_i}$$

In schema v2.0, trust was implicit and uniform (all testimony sources got the same Pramana weight regardless of source quality). This made confidence weights meaningless as a signal — a tweet and a Reuters article would get identical scores.

The fix requires explicit, per-source trust scores that reflect real-world source credibility. However, embedding trust scores directly in evidence items creates a maintenance problem: if a source's rating changes, every evidence item citing that source must be updated.

The key design insight: **source trust is a property of the source, not of individual evidence items**. It belongs in a separate registry.

## Decision

Externalise source trust into a JSONL registry at `data/registry/source_trust_registry.jsonl`. Each evidence item stores only a `source_id` key; trust is resolved from the registry at graph build time.

### Registry record schema

```json
{
  "source_id": "reuters_web_text",
  "source_name": "Reuters",
  "domain": "reuters.com",
  "source_type": "news_media",
  "modality": "web_text",
  "source_trust": 0.85,
  "prior_trust": 0.85,
  "default_evidence_types": ["testimony"],
  "default_inference_strength": 0.8,
  "trust_metadata": {
    "correct_reports": 0,
    "incorrect_reports": 0,
    "total_reports": 0,
    "last_updated": null,
    "trust_version": "v1",
    "methodology_ref": "MBFC:reuters"
  }
}
```

**Field descriptions:**

| Field | Type | Description |
|---|---|---|
| `source_id` | string | Primary key: `{domain}_{modality}` for web sources; canonical name for special sources |
| `source_name` | string | Human-readable source name |
| `domain` | string | Domain as parsed from `source_url` (without www.) |
| `source_type` | string | Category — see source type taxonomy below |
| `modality` | string | Evidence modality: `web_text`, `pdf`, `web_table`, `video`, `image`, `audio`, `simulation_state`, `annotator_knowledge`, `other` |
| `source_trust` | float | Current dynamic trust value $ST_i$ used in EC_i formula |
| `prior_trust` | float | Original prior, preserved for Bayesian recalibration and ablation |
| `default_evidence_types` | string[] | Starting evidence type labels for this source; converters apply content-based overrides |
| `default_inference_strength` | float | Baseline $IS_i$ for this source; converters override from answer content (e.g. abstractive → 0.6) |
| `trust_metadata.methodology_ref` | string | Reference to external rating used to set this score (e.g. `MBFC:reuters`, `tier1-government`) |
| `trust_metadata.resolve_original_url` | bool | For archive services only: true if the original source URL must be extracted from the archive URL |

### Source type taxonomy

| `source_type` | Description | Default $ST$ range |
|---|---|---|
| `simulation` | Closed-world simulator (AI2THOR) | 1.0 |
| `scientific_paper` | Peer-reviewed journal, NCBI/PubMed | 0.88–0.92 |
| `government` | Official government domains (.gov, .gov.uk, etc.) | 0.85–0.90 |
| `knowledge_graph` | Wikipedia, Encyclopaedia Britannica | 0.80–0.85 |
| `fact_checker` | PolitiFact, AfricaCheck, Snopes | 0.80–0.85 |
| `news_media` | Professionally edited news outlets | 0.65–0.85 |
| `ngo_or_org` | Non-government .org domains | 0.55–0.70 |
| `web_text` | General unverified web content | 0.45–0.60 |
| `web_archive` | Archive services (Wayback, archive.ph) | 0.40 (fallback) |
| `llm_generated` | Synthetic LLM-generated evidence | 0.50 |
| `social_media` | Twitter/X, Facebook, Reddit | 0.30–0.40 |
| `testimony` | Annotator knowledge without citation | 0.65 |
| `unknown` | Unrecognised source | 0.40 |

### Methodology for setting source trust scores (two-tier)

**Tier 1 — Source-type defaults:** Every `source_type` has a default trust range. These are applied when no domain-specific entry exists. The ranges are derived from the general reliability track records documented in the academic media credibility literature.

**Tier 2 — Domain-specific overrides:** For named sources in the registry, trust scores are calibrated against [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com/) factual ratings, mapped to the $[0, 1]$ scale as follows:

| MBFC Factual Rating | $ST$ range |
|---|---|
| Very High | 0.90–1.00 |
| High | 0.80–0.89 |
| Mostly Factual | 0.70–0.79 |
| Mixed | 0.55–0.69 |
| Low / Very Low | 0.30–0.54 |

Each domain-specific entry records the source of its rating in `trust_metadata.methodology_ref` (e.g. `MBFC:reuters`). Entries without a domain-specific override use `tier1-{source_type}` as the ref.

### Verifying a trust score

To verify or challenge a specific source's score:
1. Find the entry by `source_id` in `data/registry/source_trust_registry.jsonl`
2. Read `trust_metadata.methodology_ref` to identify the rating source
3. For MBFC entries: look up `https://mediabiasfactcheck.com/{slug}` (the slug follows the colon in the ref)
4. The $ST$ value should fall within the MBFC → ST mapping table above

### Hierarchical resolver (implemented in `src/core/claims/labels.py`)

Because there are ~1800 unique domains in AVeriTeC but only ~60 explicit registry entries, converters use a hierarchical lookup:

```
1. Exact match:           reuters.com + web_text   → reuters_web_text (ST=0.85)
2. Parent domain:         edition.cnn.com          → cnn.com + web_text → cnn_web_text (ST=0.75)
3. Archive extraction:    web.archive.org + URL     → extract original domain, recurse
4. TLD heuristic:         *.gov + web_text          → tld_gov_web_text (ST=0.85)
                          *.edu + pdf               → tld_edu_pdf (ST=0.88)
                          *.org + web_text          → tld_org_web_text (ST=0.60)
5. Modality fallback:     any pdf (unknown domain)  → academic_pdf (ST=0.90)
                          web_table                 → general_web_table (ST=0.60)
6. Final fallback:        unknown_web (ST=0.40)
```

TLD heuristics cover the majority of government and academic domains without requiring explicit entries for every national government TLD (.gov.za, .gov.uk, .gov.in, .gov.ng, etc.).

### Future Bayesian updates

Once the model generates predictions on real evidence, trust scores can be recalibrated using:

$$ST_{new} = \frac{\alpha + \text{correct}}{\alpha + \beta + \text{total}}$$

Where $\alpha$ and $\beta$ are the Beta distribution priors (initialised from `prior_trust`). This is tracked in `trust_metadata.correct_reports`, `incorrect_reports`, `total_reports`. Recalibration updates `source_trust` without changing `prior_trust`, preserving the original prior for ablation studies.

## Consequences

**Positive:**
- Trust is decoupled from evidence records: updating a source's trust score requires changing one registry entry, not regenerating all JSONL
- `prior_trust` field enables ablation studies that test the impact of trust scoring (set all to 0.5 → flat prior)
- Hierarchical resolver handles long-tail domains without manual enumeration
- Methodology is transparent and externally verifiable via MBFC references

**Negative:**
- Converters require a registry lookup at build time (adds one I/O dependency)
- Trust scores are static V1 values — they reflect MBFC ratings at a point in time and may become stale
- MBFC ratings cover primarily English-language media; non-English sources rely on tier-1 defaults

**Future work:**
- Integrate with NewsGuard API for automated rating updates
- Extend registry with non-English media MBFC equivalents
- Trigger Bayesian recalibration after each model evaluation run
