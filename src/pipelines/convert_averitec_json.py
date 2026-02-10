import json
import re
from datetime import datetime, timezone

LABEL_MAP = {
    "supported": "supported",
    "refuted": "refuted",
    "not enough evidence": "not_enough_evidence",
    "not_enough_evidence": "not_enough_evidence",
    "conflicting evidence/cherrypicking": "conflicting_evidence",
    "conflicting evidence": "conflicting_evidence",
    "cherrypicking": "cherrypicking",
}

ANSWER_TYPE_MAP = {
    "boolean": "boolean",
    "extractive": "extractive",
    "abstractive": "abstractive",
    "unanswerable": "unanswerable",
}

TEXTUAL_SOURCES = {"web_text", "pdf", "web_table", "other"}
PERCEPTUAL_SOURCES = {"image", "video", "audio" }

NUMERIC_CUES = re.compile(
    r"\b(%|percent|percentage|largest|smallest|rank|gdp|million|billion|trillion)\b",
    re.IGNORECASE,
)

PRIMARY_ORDER = [
    "non_apprehension",
    "perception",
    "comparison_analogy",
    "inference",
    "testimony",
    "postulation_derivation",
]

def infer_proof_types(label: str, claim_types, strategies, qa_out, evidence_items):
    """
    Returns a sorted list of proof types (multi-label).
    - Non-apprehension (Anupalabdhi)
    if label == Not Enough Evidence

    - Perception (Pratyakṣa)
    if ANY evidence medium is Image/graphic or Video (or Audio if you want)

    - Comparison/Analogy (Upamāna)
    if claim_type == Numerical Claim OR strategy includes Numerical Comparison

    - Testimony (Śabda)
    if any evidence is Web text / PDF / Web table / Metadata / Other (this will be most)

    - Inference (Anumāna)
    if number of questions ≥ 2 OR answer_type includes Abstractive AND multiple sources

    - Postulation/Derivation (Arthāpatti)
    not directly supported by Averitec fields → keep rare/empty unless you add a special detector later
    """
    proof_types = set()

    # 6) Non-apprehension (Anupalabdhi)
    if label == "not_enough_evidence":
        proof_types.add("non_apprehension")

    # Evidence modality
    src_types = {(e.get("source_type") or "").lower() for e in (evidence_items or [])}

    # 1) Perception (Pratyaksha)
    if src_types & PERCEPTUAL_SOURCES:
        proof_types.add("perception")

    # 3) Testimony (Shabda) - textual / documentary sources
    if src_types & TEXTUAL_SOURCES:
        proof_types.add("testimony")

    # 4) Comparison/Analogy (Upamana) - numeric claims/strategies
    ct = " ".join([str(c).lower() for c in (claim_types or [])])
    st = " ".join([str(s).lower() for s in (strategies or [])])

    if ("numerical claim" in ct) or ("numerical comparison" in st):
        proof_types.add("comparison_analogy")

    # numeric cue in answers (backup)
    for q in (qa_out or []):
        for a in (q.get("answers") or []):
            ans = str(a.get("answer") or "")
            if NUMERIC_CUES.search(ans):
                proof_types.add("comparison_analogy")
                break

    # 2) Inference (Anumana) - multi-Q or synthesis
    n_q = len(qa_out or [])
    if n_q >= 2:
        proof_types.add("inference")
    else:
        # single question: inference if abstractive + multiple sources
        ans_types = []
        src_urls = set()
        for q in (qa_out or []):
            for a in (q.get("answers") or []):
                ans_types.append(a.get("answer_type"))
                if a.get("source_url"):
                    src_urls.add(a.get("source_url"))
        if ("abstractive" in ans_types) and (len(src_urls) >= 2):
            proof_types.add("inference")

    # 5) Postulation/Derivation (Arthapatti) - not reliably detectable here (leave out for now)
    # proof_types.add("postulation_derivation")  # only if you later add a detector

    if not proof_types:
        # fail-safe: Averitec is mostly testimony
        proof_types.add("testimony")

    return sorted(proof_types)

def pick_primary(proof_types):
    s = set(proof_types or [])
    for p in PRIMARY_ORDER:
        if p in s:
            return p
    return proof_types[0] if proof_types else None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def normalize_label(label):
    if not label:
        return "not_enough_evidence"
    key = str(label).strip().lower()
    return LABEL_MAP.get(key, key.replace(" ", "_"))

def normalize_answer_type(t):
    if not t:
        return "unanswerable"
    key = str(t).strip().lower()
    return ANSWER_TYPE_MAP.get(key, key)

def normalize_date(d):
    """
    Converts '25-8-2020' -> '2020-08-25'
    Leaves '2020-08-25' as-is
    """
    if not d:
        return None
    d = str(d).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return d
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})", d)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    return None

def source_type_from_medium(source_medium):
    if not source_medium:
        return "other"
    sm = str(source_medium).strip().lower()
    if "web table" in sm or sm == "web_table":
        return "web_table"
    if "web text" in sm or sm == "web_text":
        return "web_text"
    if "pdf" in sm:
        return "pdf"
    if "video" in sm or "youtube" in sm:
        return "video"
    if "image" in sm or "jpeg" in sm or "png" in sm:
        return "image"
    return "other"


# def infer_proof_types(label, claim_types, strategies, qa_out, evidence_items):
#     proof_types = set()

#     # 1) non-apprehension
#     if label == "not_enough_evidence":
#         proof_types.add("non_apprehension")

#     # evidence modality
#     src_types = { (e.get("source_type") or "").lower() for e in (evidence_items or []) }

#     # 2) perception
#     if src_types & PERCEPTUAL_SOURCES:
#         proof_types.add("perception")

#     # 3) testimony (textual evidence)
#     if src_types & TEXTUAL_SOURCES:
#         proof_types.add("testimony")

#     # 4) comparison/analogy
#     ct = " ".join([str(c).lower() for c in (claim_types or [])])
#     st = " ".join([str(s).lower() for s in (strategies or [])])
#     if ("numerical" in ct) or ("numerical_comparison" in st) or ("numerical comparison" in st):
#         proof_types.add("comparison_analogy")

#     # heuristic numeric cues in answers
#     for q in (qa_out or []):
#         for a in (q.get("answers") or []):
#             ans = a.get("answer") or ""
#             if NUMERIC_CUES.search(str(ans)):
#                 proof_types.add("comparison_analogy")
#                 break

#     # 5) inference (multi-hop QA / abstractive synthesis)
#     n_q = len(qa_out or [])
#     if n_q >= 2:
#         proof_types.add("inference")
#     else:
#         # single question: still inference if abstractive + multiple sources
#         ans_types = []
#         src_urls = set()
#         for q in (qa_out or []):
#             for a in (q.get("answers") or []):
#                 ans_types.append(a.get("answer_type"))
#                 if a.get("source_url"):
#                     src_urls.add(a.get("source_url"))
#         if ("abstractive" in ans_types) and (len(src_urls) >= 2):
#             proof_types.add("inference")

#     # optional: remove testimony if only non-apprehension and no evidence
#     if (proof_types == {"non_apprehension"}) and not evidence_items:
#         pass

#     return sorted(proof_types)


# def infer_epistemic_proof_type(label, claim_types, strategies, speaker, reporting_source):
#     """
#     - testimony = if speaker OR reporting_source exists
#     - comparison_analogy = if claim is numerical OR strategy contains numerical comparison
#     - non_apprehension = if label is not_enough_evidence
#     - inference = fallback
#     """
#     if label == "not_enough_evidence":
#         return "non_apprehension"
#     ct = " ".join([str(c).lower() for c in (claim_types or [])])
#     st = " ".join([str(s).lower() for s in (strategies or [])])
#     if "numerical" in ct or "numerical comparison" in st:
#         return "comparison_analogy"
#     if speaker or reporting_source:
#         return "testimony"
#     return "inference"

# def infer_epistemic_from_evidence(label, claim_types, strategies, evidence_items):
#     if label == "not_enough_evidence":
#         return "non_apprehension"

#     # evidence modality-based perception
#     src_types = " ".join([str(e.get("source_type", "")).lower() for e in (evidence_items or [])])
#     if ("image" in src_types) or ("video" in src_types):
#         return "perception"

#     ct = " ".join([str(c).lower() for c in (claim_types or [])])
#     st = " ".join([str(s).lower() for s in (strategies or [])])

#     if "numerical" in ct or "numerical comparison" in st:
#         return "comparison_analogy"

#     # default: Averitec is mainly testimony (web/pdfs/tables)
#     return "testimony"


def to_snake_list(xs):
    if not xs:
        return None
    return [str(x).strip().lower().replace(" ", "_") for x in xs]

def convert_one(rec, rec_id):
    claim_text = (rec.get("claim") or "").strip()
    label = normalize_label(rec.get("label"))

    claim_types = rec.get("claim_types") or []
    strategies = rec.get("fact_checking_strategies") or []
    speaker = rec.get("speaker")
    reporting_source = rec.get("reporting_source")

    verdict = {
        "label": label,
        "justification": rec.get("justification"),
        "annotator_confidence": None
    }

    claim_meta = {
        "required_reannotation": rec.get("required_reannotation"),
        "claim_date": normalize_date(rec.get("claim_date")),
        "speaker": speaker,
        "original_claim_url": rec.get("original_claim_url"),
        "cached_original_claim_url": rec.get("cached_original_claim_url"),
        "fact_checking_article": rec.get("fact_checking_article"),
        "reporting_source": reporting_source,
        "location_ISO_code": rec.get("location_ISO_code"),
        "claim_types": to_snake_list(claim_types),
        "fact_checking_strategies": to_snake_list(strategies)
    }

    qa_out = []
    evidence_items = []

    questions = rec.get("questions") or []
    for qi, q in enumerate(questions, start=1):
        qtext = (q.get("question") or "").strip()
        answers_out = []
        answers = q.get("answers") or []
        for ai, a in enumerate(answers, start=1):
            evidence_id = f"{rec_id}-q{qi}-a{ai}"
            ans_type = normalize_answer_type(a.get("answer_type"))
            source_medium = a.get("source_medium")

            answers_out.append({
                "answer": str(a.get("answer", "")).strip(),
                "answer_type": ans_type,
                "source_url": a.get("source_url"),
                "cached_source_url": a.get("cached_source_url"),
                "source_medium": str(source_medium).strip().lower().replace(" ", "_") if source_medium else None,
                "boolean_explanation": a.get("boolean_explanation"),
                "evidence_ids": [evidence_id]
            })

            if label == "supported":
                stance = "supports"
            elif label == "refuted":
                stance = "refutes"
            else:
                stance = "unknown"

            evidence_items.append({
                "evidence_id": evidence_id,
                "source_type": source_type_from_medium(source_medium),
                "source_url": a.get("source_url"),
                "cached_source_url": a.get("cached_source_url"),
                "source_medium": str(source_medium).strip().lower().replace(" ", "_") if source_medium else None,
                "stance": stance,
                "extract": None,
                "evidence_triples": None
            })

        qa_out.append({"question": qtext, "answers": answers_out})
    
    # proof_type = infer_epistemic_proof_type(label, claim_types, strategies, speaker, reporting_source)
    # epistemic_type = infer_epistemic_from_evidence(label, claim_types, strategies, rec.get("evidence_items"))
    # proof_types = infer_proof_types(label, claim_types, strategies, rec.get("qa"), rec.get("evidence_items"))
    proof_types = infer_proof_types(label, claim_types, strategies, qa_out, evidence_items)
    primary = pick_primary(proof_types)

    out = {
        "id": rec.get("id") or rec_id,
        "claim": claim_text,
        "verdict": verdict,
        "epistemic": {
            "proof_types": proof_types,
            "primary_proof_type": primary,
            "proof_type_rationale": None,
            "proof_confidence": None
        },
        "claim_meta": claim_meta,
        "claim_triples": None,
        "reasoning": None,
        "qa": qa_out if qa_out else None,
        "evidence_items": evidence_items if evidence_items else None,
        "context": {
            "context_id": "schlichtkrull2023averitec",
            "context_type": "averitec",
            "generator": "averitec_converter",
            "split": rec.get("split")
        },
        "meta": {
            "created_utc": now_utc_iso(),
            "notes": None
        }
    }
    return out

def convert_file(in_path, out_path, split_name):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list at top-level.")

    unified = []
    for i, rec in enumerate(data, start=1):
        rec_id = f"averitec-{split_name}-{i:06d}"
        unified.append(convert_one(rec, rec_id))

    # Write JSONL (recommended for big datasets)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in unified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def convert_averitec_file(infile: str, outfile: str, split_name: str):
    """
    Reads Averitec JSON (top-level list) and writes unified JSONL.
    split_name: train|dev|test (used for stable ids like averitec-train-000001)
    """
    convert_file(infile, outfile, split_name)
