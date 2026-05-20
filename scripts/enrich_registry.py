"""Enrich the source trust registry with curated entries for well-known domains.

Scans AVeriTeC source URLs, finds domains still resolving to unknown_web after
TLD heuristics, and appends curated entries for known domains.

Usage:
    uv run python scripts/enrich_registry.py [--dry-run] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.epistemic.registry import (
    _normalise_domain,
    load_source_trust_registry,
    resolve_source_id,
)

REGISTRY_PATH = Path("data/registry/source_trust_registry.jsonl")
AVERITEC_PATHS = [
    Path("data/raw/averitec/train.json"),
    Path("data/raw/averitec/dev.json"),
]
_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/\d+[^/]*/(.+)")

# ── Curated domain classifications ───────────────────────────────────────────
# domain → (source_type, source_trust, display_name, methodology_ref)
KNOWN_DOMAINS: dict[str, tuple[str, float, str, str]] = {
    # Fact-checkers
    "africacheck.org":               ("fact_checker",    0.80, "Africa Check",                      "MBFC:africa-check"),
    "factcheck.org":                 ("fact_checker",    0.82, "FactCheck.org",                     "MBFC:factcheck-org"),
    "factcheck.afp.com":             ("fact_checker",    0.82, "AFP Fact Check",                    "MBFC:afp-fact-check"),
    "altnews.in":                    ("fact_checker",    0.78, "Alt News",                          "MBFC:alt-news"),
    "fullfact.org":                  ("fact_checker",    0.82, "Full Fact",                         "MBFC:full-fact"),
    "snopes.com":                    ("fact_checker",    0.75, "Snopes",                            "MBFC:snopes"),
    "politifact.com":                ("fact_checker",    0.80, "PolitiFact",                        "MBFC:politifact"),
    "boomlive.in":                   ("fact_checker",    0.78, "Boom Live",                         "MBFC:boomlive"),
    "vishvasnews.com":               ("fact_checker",    0.72, "Vishvas News",                      "MBFC:vishvas-news"),
    "logically.ai":                  ("fact_checker",    0.75, "Logically",                         "MBFC:logically"),
    "checkyourfact.com":             ("fact_checker",    0.72, "Check Your Fact",                   "MBFC:check-your-fact"),
    # News — US
    "wsj.com":                       ("news_media",      0.75, "Wall Street Journal",               "MBFC:wsj"),
    "foxnews.com":                   ("news_media",      0.55, "Fox News",                          "MBFC:fox-news"),
    "abcnews.go.com":                ("news_media",      0.72, "ABC News",                          "MBFC:abc-news"),
    "thehill.com":                   ("news_media",      0.70, "The Hill",                          "MBFC:the-hill"),
    "huffpost.com":                  ("news_media",      0.60, "HuffPost",                          "MBFC:huffpost"),
    "thedailybeast.com":             ("news_media",      0.62, "The Daily Beast",                   "MBFC:daily-beast"),
    "msnbc.com":                     ("news_media",      0.60, "MSNBC",                             "MBFC:msnbc"),
    "vice.com":                      ("news_media",      0.65, "VICE",                              "MBFC:vice"),
    "vox.com":                       ("news_media",      0.68, "Vox",                               "MBFC:vox"),
    "slate.com":                     ("news_media",      0.65, "Slate",                             "MBFC:slate"),
    "nbcnews.com":                   ("news_media",      0.72, "NBC News",                          "MBFC:nbc-news"),
    "cbsnews.com":                   ("news_media",      0.72, "CBS News",                          "MBFC:cbs-news"),
    "usatoday.com":                  ("news_media",      0.70, "USA Today",                         "MBFC:usa-today"),
    "latimes.com":                   ("news_media",      0.72, "Los Angeles Times",                 "MBFC:la-times"),
    "newsweek.com":                  ("news_media",      0.65, "Newsweek",                          "MBFC:newsweek"),
    "theatlantic.com":               ("news_media",      0.75, "The Atlantic",                      "MBFC:the-atlantic"),
    "axios.com":                     ("news_media",      0.72, "Axios",                             "MBFC:axios"),
    "bloomberg.com":                 ("news_media",      0.78, "Bloomberg",                         "MBFC:bloomberg"),
    "businessinsider.com":           ("news_media",      0.65, "Business Insider",                  "MBFC:business-insider"),
    "npr.org":                       ("news_media",      0.80, "NPR",                               "MBFC:npr"),
    "propublica.org":                ("news_media",      0.82, "ProPublica",                        "MBFC:propublica"),
    "theintercept.com":              ("news_media",      0.65, "The Intercept",                     "MBFC:the-intercept"),
    # News — International
    "aljazeera.com":                 ("news_media",      0.72, "Al Jazeera",                        "MBFC:aljazeera"),
    "independent.co.uk":             ("news_media",      0.65, "The Independent",                   "MBFC:the-independent"),
    "thetimes.co.uk":                ("news_media",      0.72, "The Times (UK)",                    "MBFC:the-times"),
    "telegraph.co.uk":               ("news_media",      0.65, "The Telegraph",                     "MBFC:telegraph"),
    "mirror.co.uk":                  ("news_media",      0.50, "Daily Mirror",                      "MBFC:daily-mirror"),
    "thesun.co.uk":                  ("news_media",      0.45, "The Sun (UK)",                      "MBFC:the-sun"),
    "dailymail.co.uk":               ("news_media",      0.50, "Daily Mail",                        "MBFC:daily-mail"),
    "timesofindia.indiatimes.com":   ("news_media",      0.68, "Times of India",                    "MBFC:times-of-india"),
    "theprint.in":                   ("news_media",      0.68, "The Print",                         "MBFC:the-print"),
    "ndtv.com":                      ("news_media",      0.68, "NDTV",                              "MBFC:ndtv"),
    "scroll.in":                     ("news_media",      0.68, "Scroll.in",                         "MBFC:scroll-in"),
    "thewire.in":                    ("news_media",      0.68, "The Wire",                          "MBFC:the-wire"),
    "hindustantimes.com":            ("news_media",      0.68, "Hindustan Times",                   "MBFC:hindustan-times"),
    "firstpost.com":                 ("news_media",      0.62, "Firstpost",                         "MBFC:firstpost"),
    "opindia.com":                   ("news_media",      0.40, "OpIndia",                           "MBFC:opindia"),
    # Scientific / Academic
    "nature.com":                    ("scientific_paper", 0.90, "Nature",                           "tier1-academic"),
    "thelancet.com":                 ("scientific_paper", 0.90, "The Lancet",                       "tier1-academic"),
    "bmj.com":                       ("scientific_paper", 0.90, "BMJ",                              "tier1-academic"),
    "pnas.org":                      ("scientific_paper", 0.88, "PNAS",                             "tier1-academic"),
    "jstor.org":                     ("scientific_paper", 0.85, "JSTOR",                            "tier1-academic"),
    "sciencedirect.com":             ("scientific_paper", 0.85, "ScienceDirect",                    "tier1-academic"),
    "springer.com":                  ("scientific_paper", 0.85, "Springer",                         "tier1-academic"),
    "tandfonline.com":               ("scientific_paper", 0.85, "Taylor & Francis Online",          "tier1-academic"),
    "jamanetwork.com":               ("scientific_paper", 0.90, "JAMA Network",                     "tier1-academic"),
    "nejm.org":                      ("scientific_paper", 0.90, "New England Journal of Medicine",  "tier1-academic"),
    "cell.com":                      ("scientific_paper", 0.90, "Cell Press",                       "tier1-academic"),
    "medrxiv.org":                   ("scientific_paper", 0.80, "medRxiv (preprint)",                "tier1-academic"),
    "biorxiv.org":                   ("scientific_paper", 0.78, "bioRxiv (preprint)",                "tier1-academic"),
    # Government / IGO
    "who.int":                       ("government",      0.85, "World Health Organization",         "tier1-government"),
    "imf.org":                       ("government",      0.82, "International Monetary Fund",       "tier1-government"),
    "ec.europa.eu":                  ("government",      0.82, "European Commission",               "tier1-government"),
    "population.un.org":             ("government",      0.85, "UN Population Division",            "tier1-government"),
    "fred.stlouisfed.org":           ("government",      0.85, "FRED (St. Louis Fed)",              "tier1-government"),
    "data.worldbank.org":            ("government",      0.85, "World Bank Data",                   "tier1-government"),
    "stats.oecd.org":                ("government",      0.85, "OECD Statistics",                   "tier1-government"),
    "knbs.or.ke":                    ("government",      0.82, "Kenya National Bureau of Statistics","tier1-government"),
    "idea.int":                      ("government",      0.80, "International IDEA",                "tier1-government"),
    "ons.gov.uk":                    ("government",      0.88, "UK Office for National Statistics",  "tier1-government"),
    "niti.gov.in":                   ("government",      0.82, "NITI Aayog",                        "tier1-government"),
    "iaea.org":                      ("government",      0.82, "IAEA",                              "tier1-government"),
    "unwomen.org":                   ("government",      0.82, "UN Women",                          "tier1-government"),
    # Knowledge graph / reference
    "govtrack.us":                   ("knowledge_graph", 0.75, "GovTrack",                          "tier1-knowledge-graph"),
    "ourworldindata.org":            ("knowledge_graph", 0.82, "Our World in Data",                 "tier1-knowledge-graph"),
    "statista.com":                  ("knowledge_graph", 0.70, "Statista",                          "tier1-knowledge-graph"),
    "tradingeconomics.com":          ("knowledge_graph", 0.68, "Trading Economics",                 "tier1-knowledge-graph"),
    "macrotrends.net":               ("knowledge_graph", 0.68, "Macrotrends",                       "tier1-knowledge-graph"),
    # Health information (non-academic)
    "mayoclinic.org":                ("web_text",        0.82, "Mayo Clinic",                       "tier1-health"),
    "clevelandclinic.org":           ("web_text",        0.72, "Cleveland Clinic",                  "tier1-health"),
    "medicalnewstoday.com":          ("web_text",        0.62, "Medical News Today",                "tier1-health"),
    "healthline.com":                ("web_text",        0.62, "Healthline",                        "tier1-health"),
    # Web archive
    "archive.ph":                    ("web_archive",     0.40, "Archive.ph / Archive.today",        "tier1-web-archive"),
    # General web
    "medium.com":                    ("web_text",        0.40, "Medium",                            "tier1-web-text"),
    "rev.com":                       ("web_text",        0.40, "Rev",                               "tier1-web-text"),
    "substack.com":                  ("web_text",        0.40, "Substack",                          "tier1-web-text"),
    "quora.com":                     ("web_text",        0.35, "Quora",                             "tier1-web-text"),
    "scribd.com":                    ("web_text",        0.40, "Scribd",                            "tier1-web-text"),
}

_DEFAULT_IS: dict[str, float] = {
    "fact_checker":    0.75,
    "news_media":      0.70,
    "scientific_paper": 0.75,
    "government":      0.80,
    "knowledge_graph": 0.80,
    "web_archive":     0.70,
    "web_text":        0.70,
    "ngo_or_org":      0.70,
}

_DEFAULT_EV_TYPES: dict[str, list[str]] = {
    "fact_checker":    ["testimony"],
    "news_media":      ["testimony"],
    "scientific_paper": ["testimony", "inference"],
    "government":      ["testimony"],
    "knowledge_graph": ["testimony"],
    "web_archive":     ["testimony"],
    "web_text":        ["testimony"],
    "ngo_or_org":      ["testimony"],
}


def _extract_domain(url: str) -> str | None:
    """Extract domain from URL, unwrapping Wayback Machine archives."""
    if not url:
        return None
    url = url.strip()
    if url.lower() == "metadata":
        return None
    m = _ARCHIVE_RE.search(url)
    if m:
        embedded = m.group(1)
        if not embedded.startswith(("http://", "https://")):
            embedded = "https://" + embedded
        url = embedded
    parsed = urlparse(url if "://" in url else "https://" + url)
    domain = (parsed.netloc or "").lower().removeprefix("www.")
    return domain or None


def _scan_averitec_domains() -> list[str]:
    """Collect all unique source URL domains from AVeriTeC raw files."""
    domains: set[str] = set()
    for path in AVERITEC_PATHS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            for answer in item.get("questions", []):
                for src in answer.get("answers", []):
                    d = _extract_domain(src.get("source_url", ""))
                    if d:
                        domains.add(d)
    return sorted(domains)


def _build_entry(domain: str, source_type: str, trust: float, name: str, ref: str, modality: str = "web_text") -> dict:
    """Build a registry entry dict for a domain."""
    ev_types = _DEFAULT_EV_TYPES.get(source_type, ["testimony"])
    default_is = _DEFAULT_IS.get(source_type, 0.70)
    source_id = f"{_normalise_domain(domain)}_{modality}"
    return {
        "source_id": source_id,
        "source_name": name,
        "domain": domain,
        "source_type": source_type,
        "modality": modality,
        "source_trust": trust,
        "prior_trust": trust,
        "default_evidence_types": ev_types,
        "default_inference_strength": default_is,
        "trust_metadata": {
            "correct_reports": 0,
            "incorrect_reports": 0,
            "total_reports": 0,
            "last_updated": None,
            "trust_version": "v1",
            "methodology_ref": ref,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print new entries without writing")
    parser.add_argument("--out", default=str(REGISTRY_PATH), help="Output registry path")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    registry = load_source_trust_registry(out_path)

    all_domains = _scan_averitec_domains()
    print(f"Found {len(all_domains)} unique domains in AVeriTeC data")

    unknown_domains = [d for d in all_domains if resolve_source_id(d, "web_text", registry) == "unknown_web"]
    print(f"  {len(unknown_domains)} still resolve to unknown_web after TLD heuristics")

    new_entries: list[dict] = []
    seen_sids: set[str] = set()
    skipped_already_known: list[str] = []
    skipped_not_in_curated: list[str] = []

    for domain in sorted(KNOWN_DOMAINS):
        source_type, trust, name, ref = KNOWN_DOMAINS[domain]
        entry = _build_entry(domain, source_type, trust, name, ref, "web_text")
        sid = entry["source_id"]

        if sid in registry or sid in seen_sids:
            skipped_already_known.append(domain)
            continue

        # Check if it already resolves to something non-unknown via TLD heuristic
        current = resolve_source_id(domain, "web_text", registry)
        if current != "unknown_web":
            skipped_already_known.append(domain)
            continue

        new_entries.append(entry)
        seen_sids.add(sid)

        # Also add PDF entry for academic + government sources
        if source_type in ("scientific_paper", "government"):
            pdf_entry = _build_entry(domain, source_type, trust, name, ref, "pdf")
            pdf_sid = pdf_entry["source_id"]
            if pdf_sid not in registry and pdf_sid not in seen_sids:
                new_entries.append(pdf_entry)
                seen_sids.add(pdf_sid)

    # Report unknowns not in curated list
    curated_domains = set(KNOWN_DOMAINS.keys())
    for d in unknown_domains:
        # Check if domain is covered by any KNOWN_DOMAINS entry (exact or parent)
        covered = d in curated_domains or _strip_www(d) in curated_domains
        if not covered:
            skipped_not_in_curated.append(d)

    print(f"\n{'=' * 60}")
    print(f"  New entries to add:         {len(new_entries)}")
    print(f"  Already registered/covered: {len(skipped_already_known)}")
    print(f"  Not in curated list:        {len(skipped_not_in_curated)}")
    print(f"{'=' * 60}")

    if new_entries:
        print("\nNew entries:")
        for e in new_entries:
            print(f"  [{e['source_type']:18}] {e['source_id']:45} ST={e['source_trust']:.2f}")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    with open(out_path, "a", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nAppended {len(new_entries)} entries to {out_path}")


def _strip_www(domain: str) -> str:
    return domain.removeprefix("www.")


if __name__ == "__main__":
    main()
