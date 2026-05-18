"""
Tests that all source_ids used in raw data are present in the source trust registry,
and that all AVeriTeC URLs resolve to a known registry entry.
"""

import json
import collections
from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.epistemic.registry import load_source_trust_registry, resolve_source_id, get_source_trust

ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "data/registry/source_trust_registry.jsonl"
RAW_AI2THOR   = ROOT / "data/raw/ai2thor/claims_all.jsonl"
RAW_SYNTHETIC = ROOT / "data/raw/synthetic/synthetic_current.jsonl"
AVERITEC_SPLITS = [
    ROOT / "data/raw/averitec/train.json",
    ROOT / "data/raw/averitec/dev.json",
    ROOT / "data/raw/averitec/test.json",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry() -> dict:
    return load_source_trust_registry(str(REGISTRY_PATH))


@pytest.fixture(scope="module")
def explicit_source_ids(registry) -> dict[str, set]:
    """Collect every explicit source_id from JSONL data files → {source_id: {filename}}."""
    found: dict[str, set] = collections.defaultdict(set)
    for path in (RAW_AI2THOR, RAW_SYNTHETIC):
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            for ev in json.loads(line).get("evidence", []):
                sid = ev.get("source_id")
                if sid:
                    found[sid].add(path.name)
    return dict(found)


@pytest.fixture(scope="module")
def averitec_resolved_ids(registry) -> set[str]:
    """Resolve every AVeriTeC source_url to a registry source_id."""
    resolved: set[str] = set()
    for split_path in AVERITEC_SPLITS:
        if not split_path.exists():
            continue
        data = json.loads(split_path.read_text("utf-8"))
        for rec in data:
            for q in rec.get("questions", []):
                for a in q.get("answers", []):
                    url = a.get("source_url", "")
                    if not url:
                        continue
                    parsed = urlparse(url if "://" in url else "https://" + url)
                    domain = parsed.netloc or parsed.path.split("/")[0]
                    domain = domain.lower().removeprefix("www.")
                    if not domain:
                        continue
                    modality = "pdf" if url.lower().endswith(".pdf") else "web_text"
                    resolved.add(resolve_source_id(domain, modality, registry))
    return resolved


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExplicitSourceIds:
    """Explicit source_ids written into JSONL evidence records."""

    def test_all_explicit_ids_in_registry(self, explicit_source_ids, registry):
        missing = {sid for sid in explicit_source_ids if sid not in registry}
        assert not missing, (
            f"source_ids found in data but missing from registry: {sorted(missing)}"
        )

    def test_explicit_ids_have_positive_trust(self, explicit_source_ids, registry):
        for sid in explicit_source_ids:
            st = get_source_trust(sid, registry)
            assert st > 0, f"source_id '{sid}' has non-positive source_trust={st}"


class TestAveritecResolution:
    """Every AVeriTeC URL resolves to a registry entry (no silent DEFAULT fallback)."""

    def test_all_averitec_resolved_ids_in_registry(self, averitec_resolved_ids, registry):
        missing = averitec_resolved_ids - set(registry.keys())
        assert not missing, (
            f"AVeriTeC URLs resolved to source_ids missing from registry: {sorted(missing)}"
        )

    def test_averitec_resolves_at_least_one_entry(self, averitec_resolved_ids):
        assert len(averitec_resolved_ids) > 0, (
            "No AVeriTeC URLs were resolved — check that source_url fields exist in raw data"
        )

    def test_averitec_resolved_ids_have_positive_trust(self, averitec_resolved_ids, registry):
        for sid in averitec_resolved_ids:
            st = get_source_trust(sid, registry)
            assert st > 0, f"Resolved source_id '{sid}' has non-positive source_trust={st}"
