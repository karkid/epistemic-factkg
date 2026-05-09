import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x else None
    return x


def main():
    ap = argparse.ArgumentParser(
        description="Analyze raw Averitec JSON and collect unique values + counts."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more Averitec JSON files (train/dev/test).",
    )
    ap.add_argument(
        "--out", default="averitec_profile.json", help="Output JSON profile file"
    )
    args = ap.parse_args()

    # Global counters
    label_counts = Counter()
    claim_types_counts = Counter()
    strategies_counts = Counter()

    # QA-level
    answer_type_counts = Counter()
    source_medium_counts = Counter()
    has_boolean_expl_counts = Counter()

    # Evidence modality (derived later from source_medium)
    derived_source_type_counts = Counter()

    # Missingness / meta stats
    missing = Counter()  # counts of missing fields across claims
    speaker_counts = Counter({"present": 0, "missing": 0})
    reporting_source_counts = Counter({"present": 0, "missing": 0})
    original_url_counts = Counter({"present": 0, "missing": 0})
    fact_check_article_counts = Counter({"present": 0, "missing": 0})
    location_counts = Counter()

    # Collect unique sets (for debugging / rule design)
    unique = defaultdict(set)

    def source_type_from_medium(source_medium: str) -> str:
        if not source_medium:
            return "other"
        sm = source_medium.strip().lower()
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

    total_claims = 0
    total_questions = 0
    total_answers = 0

    for inp in args.inputs:
        data = load_json(inp)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level list in {inp}")

        for rec in data:
            total_claims += 1

            # Label
            label = norm(rec.get("label"))
            if label is not None:
                label_counts[label] += 1
                unique["label"].add(label)
            else:
                missing["label"] += 1

            # Meta fields presence
            speaker = norm(rec.get("speaker"))
            speaker_counts["present" if speaker else "missing"] += 1
            if speaker:
                unique["speaker"].add(speaker)

            rs = norm(rec.get("reporting_source"))
            reporting_source_counts["present" if rs else "missing"] += 1
            if rs:
                unique["reporting_source"].add(rs)

            ou = norm(rec.get("original_claim_url"))
            original_url_counts["present" if ou else "missing"] += 1

            fca = norm(rec.get("fact_checking_article"))
            fact_check_article_counts["present" if fca else "missing"] += 1

            loc = norm(rec.get("location_ISO_code"))
            if loc:
                location_counts[loc] += 1
                unique["location_ISO_code"].add(loc)
            else:
                missing["location_ISO_code"] += 1

            # claim_types
            cts = rec.get("claim_types") or []
            if not cts:
                missing["claim_types"] += 1
            for ct in cts:
                ct = norm(ct)
                if ct:
                    claim_types_counts[ct] += 1
                    unique["claim_types"].add(ct)

            # fact_checking_strategies
            fcs = rec.get("fact_checking_strategies") or []
            if not fcs:
                missing["fact_checking_strategies"] += 1
            for s in fcs:
                s = norm(s)
                if s:
                    strategies_counts[s] += 1
                    unique["fact_checking_strategies"].add(s)

            # Questions / answers
            questions = rec.get("questions") or []
            if not questions:
                missing["questions"] += 1
            for q in questions:
                total_questions += 1
                qtext = norm((q or {}).get("question"))
                if qtext:
                    unique["question_text_examples"].add(qtext[:120])

                answers = (q or {}).get("answers") or []
                if not answers:
                    missing["answers"] += 1
                for a in answers:
                    total_answers += 1

                    at = norm((a or {}).get("answer_type"))
                    if at:
                        answer_type_counts[at] += 1
                        unique["answer_type"].add(at)
                    else:
                        missing["answer_type"] += 1

                    sm = norm((a or {}).get("source_medium"))
                    if sm:
                        source_medium_counts[sm] += 1
                        unique["source_medium"].add(sm)
                    else:
                        missing["source_medium"] += 1

                    derived_source_type_counts[source_type_from_medium(sm or "")] += 1

                    be = (a or {}).get("boolean_explanation")
                    has_boolean_expl_counts["present" if norm(be) else "missing"] += 1

    # Build output
    out = {
        "inputs": args.inputs,
        "totals": {
            "claims": total_claims,
            "questions": total_questions,
            "answers": total_answers,
        },
        "counts": {
            "label": dict(label_counts.most_common()),
            "claim_types": dict(claim_types_counts.most_common()),
            "fact_checking_strategies": dict(strategies_counts.most_common()),
            "answer_type": dict(answer_type_counts.most_common()),
            "source_medium": dict(source_medium_counts.most_common()),
            "derived_source_type": dict(derived_source_type_counts.most_common()),
            "location_ISO_code": dict(location_counts.most_common()),
            "presence": {
                "speaker": dict(speaker_counts),
                "reporting_source": dict(reporting_source_counts),
                "original_claim_url": dict(original_url_counts),
                "fact_checking_article": dict(fact_check_article_counts),
                "boolean_explanation": dict(has_boolean_expl_counts),
            },
            "missing_fields": dict(missing.most_common()),
        },
        "unique_values": {
            # Keep these capped so output doesn't explode
            "label": sorted(list(unique["label"])),
            "answer_type": sorted(list(unique["answer_type"])),
            "source_medium": sorted(list(unique["source_medium"])),
            "claim_types": sorted(list(unique["claim_types"])),
            "fact_checking_strategies": sorted(
                list(unique["fact_checking_strategies"])
            ),
            "location_ISO_code": sorted(list(unique["location_ISO_code"])),
            "reporting_source_examples_top50": sorted(list(unique["reporting_source"]))[
                :50
            ],
            "speaker_examples_top50": sorted(list(unique["speaker"]))[:50],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote profile: {out_path}")
    print(
        f"Claims: {total_claims} | Questions: {total_questions} | Answers: {total_answers}"
    )
    print("\nTop derived_source_type:")
    for k, v in derived_source_type_counts.most_common(10):
        print(f"  {k}: {v}")
    print("\nTop answer_type:")
    for k, v in answer_type_counts.most_common(10):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
