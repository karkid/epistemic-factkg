# ADR-020: Webarchive URL Source Trust Resolution

**Status:** Accepted  
**Date:** 2026-05-16  
**Builds on:** ADR-009 (source trust registry), ADR-011 (evidence labeling rules)

---

## Context

AVeriTeC evidence frequently links to Wayback Machine (web.archive.org) snapshots
rather than live URLs. The original `resolve_source_id` function detected the
`web.archive.org` domain and returned `"webarchive_web_text"` (ST = 0.40) for all
such evidence.

Quantified impact:

| source_id            | ST   | Share of AVeriTeC evidence |
|----------------------|------|---------------------------|
| webarchive_web_text  | 0.40 | **34.8%**                 |
| unknown_web          | 0.40 | 31.9%                     |

**66.7% of AVeriTeC evidence was receiving ST = 0.40**, including archived Reuters,
BBC, CDC, and government pages which should receive ST ≥ 0.80.

This poisoned the EC formula: `EC = 1 - (1-ST)^(EW×IS)` with ST = 0.40 instead
of ST = 0.85 more than halves EC, biasing the SymbolicAggregator toward NEI.
This explained ~5.5 percentage point deficit for v1-hgnn on AVeriTeC vs baseline
(which bypasses EC entirely).

---

## Decision

In `AveritecConverter._resolve_evidence_source`, extract the original URL from
Wayback Machine paths before calling `resolve_source_id`:

```python
_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/\d+[^/]*/(.+)")

m = _ARCHIVE_RE.search(source_url)
if m:
    embedded = m.group(1)
    if not embedded.startswith(("http://", "https://")):
        embedded = "https://" + embedded
    source_url = embedded
```

The extracted URL is then resolved normally through the registry lookup chain.
Social media URLs embedded in archives (twitter.com, facebook.com) correctly
retain their low ST scores after extraction.

---

## Alternatives Considered

**A. Assign webarchive a higher trust score** — blanket trust elevation is wrong;
archived pages may be from any source. Rejected.

**B. Drop webarchive evidence** — loses 34.8% of AVeriTeC evidence. Rejected.

**C. Maintain a domain-to-trust mapping for common archive sources** — brittle
and incomplete. The URL extraction approach is principled and exhaustive.

---

## Consequences

- 34.8% of AVeriTeC evidence moves from ST = 0.40 to source-accurate ST (0.30–0.88
  depending on the archived domain).
- The IS distribution shifts toward correct ground-truth IS values.
- Dataset must be rebuilt after this fix. Models trained on the old data have
  systematically wrong EC values for webarchive evidence.
- The `"webarchive_web_text"` source_id becomes a rare fallback for malformed
  archive URLs only.
