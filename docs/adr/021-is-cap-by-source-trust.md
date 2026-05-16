# ADR-021: Inference Strength Capped by Source Trust

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-010 (per-evidence IS), ADR-009 (source trust registry), ADR-019 (IS rubric)

---

## Context

AVeriTeC annotators accept evidence from any web source — including social media —
as sufficient to label a claim "Supported" or "Refuted". The IS rubric (ADR-019)
assigns IS = 0.8 to extractive answers regardless of source.

This creates an epistemic contradiction: a Facebook post with ST = 0.30 cannot
legitimately provide IS = 0.8 inference strength. The Shabda (testimony) Pramana
is only as strong as the trustworthiness of the testifier. High IS from a low-trust
source is an oxymoron in the framework.

In the EC formula `EC = 1 - (1-ST)^(EW×IS)`, even with ST = 0.30 the IS value
inflates EC beyond what the source credibility warrants.

---

## Decision

Cap IS by source trust for sources below a trust threshold:

```python
ST_THRESHOLD = 0.45
is_ = is_raw if st >= ST_THRESHOLD else max(0.10, min(is_raw, st))
```

Effect by source:

| source_id          | ST   | IS_raw | IS_final |
|--------------------|------|--------|----------|
| reuters_web_text   | 0.85 | 0.80   | 0.80     |
| unknown_web        | 0.40 | 0.80   | 0.40     |
| twitter_web_text   | 0.35 | 0.80   | 0.35     |
| facebook_web_text  | 0.30 | 0.80   | 0.30     |

The ST_THRESHOLD of 0.45 captures the "unverifiable web" tier (unknown_web,
social media) while leaving news, government, and academic sources uncapped.

---

## Alternatives Considered

**A. IS = ST × IS_raw (multiplicative scaling)** — creates a smoother curve but
reduces IS for trusted sources too. Rejected; trusted sources should not be penalised.

**B. Filter out low-trust-source evidence entirely** — loses 30%+ of AVeriTeC
evidence. Rejected; the evidence still provides signal, just weaker signal.

**C. Accept the mismatch as a data limitation** — acknowledged in the paper as
a known property of crowdsourced annotations (they conflate source trust and claim
truth). The cap is applied to give the model epistemically consistent training
targets.

---

## Consequences

- IS mean in the training set shifts from ~0.73 to ~0.63 after this fix.
- For claims where AVeriTeC says "Supported" based solely on a tweet, the model
  now learns a low EC and may correctly predict NEI — diverging from the AVeriTeC
  label. This is epistemically correct and is reported as a finding in the paper.
- Models must be retrained after this fix. The IS distribution shift affects
  EC calibration and requires re-learning VerdictHead thresholds.
