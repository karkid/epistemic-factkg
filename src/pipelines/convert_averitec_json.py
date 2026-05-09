"""
Thin wrapper — real logic lives in src/adapters/averitec/converter.py.
Kept for backward compatibility with src/cli/convert_to_unified.py.

infer_proof_types / pick_primary are retained for any external callers.
"""

from src.adapters.averitec.converter import (
    AveritecConverter,
    _infer_pramana,
    _normalize_label,
    _medium_to_modality,
    _PRIMARY_ORDER,
)

_converter = AveritecConverter()


# ---------------------------------------------------------------------------
# Legacy helpers retained for backward compatibility
# ---------------------------------------------------------------------------


def pick_primary(proof_types):
    s = set(proof_types or [])
    for p in _PRIMARY_ORDER:
        if p in s:
            return p
    return proof_types[0] if proof_types else None


def infer_proof_types(label: str, claim_types, strategies, qa_out, evidence_items):
    """
    Legacy signature wrapper. Delegates to the new pramana inference logic.
    """
    label_norm = _normalize_label(label)

    modalities: set[str] = set()
    src_urls: set[str] = set()
    answer_types: list[str] = []
    answers_text_parts: list[str] = []

    for e in evidence_items or []:
        mod = _medium_to_modality(e.get("source_medium") or e.get("source_type"))
        modalities.add(mod)
        if e.get("source_url"):
            src_urls.add(e["source_url"])

    for q in qa_out or []:
        for a in q.get("answers") or []:
            at = str(a.get("answer_type") or "").strip().lower()
            answer_types.append(at)
            answers_text_parts.append(str(a.get("answer") or ""))
            if a.get("source_url"):
                src_urls.add(a["source_url"])

    # numeric cues from claim_types / strategies as before
    ct = " ".join(str(c).lower() for c in (claim_types or []))
    st = " ".join(str(s).lower() for s in (strategies or []))
    extra_text = f"{ct} {st} " + " ".join(answers_text_parts)

    _, sorted_types, _ = _infer_pramana(
        label_norm, modalities, src_urls, answer_types, extra_text
    )
    return sorted_types


def convert_averitec_file(infile: str, outfile: str, split_name: str) -> int:
    return _converter.convert_file(infile, outfile, split=split_name)
