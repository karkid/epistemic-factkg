from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple

from src.utils.io import write_jsonl
from src.schema.schema_defs import make_record
from src.generators.rdf.ttl_loader import load_ttl_triples, group_by_floorplan
from src.generators.rdf.claim_maker import verbalize
from src.generators.rdf.corruptions import corrupt_triple, is_boolean_predicate
from src.generators.rdf.claim_maker import is_verbalizable_predicate


Triple = Tuple[str, str, str]

def infer_structural_reasoning(claim: str, evidence_triples: List[Triple]) -> str:
    c = claim.lower()
    if " and " in c:
        return "conjunction"
    if " not " in f" {c} ":
        return "negation"
    # existence not used in this first generator
    return "one-hop" if len(evidence_triples) == 1 else "conjunction"


def make_conjunction_claim(c1: str, c2: str) -> str:
    # Ensure clean join
    c1 = c1.strip().rstrip(".")
    c2 = c2.strip()
    # lowercase first letter of second sentence
    if c2:
        c2 = c2[0].lower() + c2[1:]
    return f"{c1} and {c2}"


def generate_one_hop(
    floorplan_id: str,
    triples: List[Triple],
    triple_set: Set[Triple],
    n_supported: int,
    n_refuted: int,
    seed_prefix: str,
) -> List[Dict[str, Any]]:
    rows = []
    attempts = 0
    max_attempts = n_supported * 10

    # -------- SUPPORTED quota fill --------
    sup_count = 0
    while sup_count < n_supported and attempts < max_attempts:
        t = random.choice(triples)
        claim = verbalize(t)

        if not claim.strip():
            attempts += 1
            continue

        rid = f"{seed_prefix}-{floorplan_id}-onehop-sup-{sup_count:06d}"
        rows.append(
            make_record(
                rec_id=rid,
                claim=claim,
                label="SUPPORTED",
                structural_reasoning="one-hop",
                evidence_type="Perception",
                evidence_triples=[t],
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes=None,
            )
        )

        sup_count += 1
        attempts += 1

    # -------- REFUTED quota fill --------
    ref_count = 0
    attempts = 0
    max_attempts = n_refuted * 15

    while ref_count < n_refuted and attempts < max_attempts:
        t = random.choice(triples)
        tb = corrupt_triple(t, triples, triple_set)

        claim = verbalize(tb)
        if not claim.strip():
            attempts += 1
            continue

        if tb in triple_set:
            attempts += 1
            continue

        rid = f"{seed_prefix}-{floorplan_id}-onehop-ref-{ref_count:06d}"
        rows.append(
            make_record(
                rec_id=rid,
                claim=claim,
                label="REFUTED",
                structural_reasoning="one-hop",
                evidence_type="Perception",
                evidence_triples=[tb],
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes="corrupted",
            )
        )

        ref_count += 1
        attempts += 1

    return rows


def generate_negation_pairs(
    floorplan_id: str,
    triples: List[Triple],
    triple_set: Set[Triple],
    max_pairs: int,
    seed_prefix: str,
) -> List[Dict[str, Any]]:
    """
    For boolean predicates, generate both forms:
      if (s,isOpen,False) exists:
         "X is open." -> REFUTED
         "X is not open." -> SUPPORTED
    We do this by verbalizing both the true triple and its flipped version.
    """
    rows: List[Dict[str, Any]] = []

    bool_triples = [t for t in triples if is_boolean_predicate(t[1])]
    if not bool_triples:
        return rows

    sample = random.sample(bool_triples, k=min(max_pairs, len(bool_triples)))

    for idx, t in enumerate(sample):
        s, p, o = t
        # true claim
        claim_true = verbalize(t)
        if claim_true is None or claim_true.strip() == "":
            continue  # skip if verbalization fails or is empty

        # flipped
        flipped_o = "True" if str(o).lower() == "false" else "False"
        t_flip = (s, p, flipped_o)
        claim_flip = verbalize(t_flip)
        if claim_flip is None or claim_flip.strip() == "":
            continue  # skip if verbalization fails or is empty

        # Determine which one is supported by checking truth set
        true_supported = t in triple_set
        flip_supported = t_flip in triple_set

        # Assign labels
        lab_true = "SUPPORTED" if true_supported else "REFUTED"
        lab_flip = "SUPPORTED" if flip_supported else "REFUTED"

        rid1 = f"{seed_prefix}-{floorplan_id}-neg-a-{idx:06d}"
        rid2 = f"{seed_prefix}-{floorplan_id}-neg-b-{idx:06d}"

        rows.append(
            make_record(
                rec_id=rid1,
                claim=claim_true,
                label=lab_true,
                structural_reasoning="negation",
                evidence_type="Perception",
                evidence_triples=[t],
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes="negation_pair",
            )
        )
        rows.append(
            make_record(
                rec_id=rid2,
                claim=claim_flip,
                label=lab_flip,
                structural_reasoning="negation",
                evidence_type="Perception",
                evidence_triples=[t_flip],
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes="negation_pair_flipped",
            )
        )

    return rows


def generate_conjunction(
    floorplan_id: str,
    supported_onehop: List[Dict[str, Any]],
    triples: List[Triple],
    triple_set: Set[Triple],
    n_supported: int,
    n_refuted: int,
    seed_prefix: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Grab supported one-hop items to combine
    candidates = [r for r in supported_onehop if r.get("label") == "SUPPORTED"]
    if len(candidates) < 2:
        return rows

    random.shuffle(candidates)

    # Supported conjunctions: combine two supported facts
    for idx in range(min(n_supported, len(candidates) // 2)):
        a = candidates[2 * idx]
        b = candidates[2 * idx + 1]
        claim = make_conjunction_claim(a["claim"], b["claim"])
        ev = [
            tuple(a["evidence"]["triples"][0]),
            tuple(b["evidence"]["triples"][0]),
        ]
        rid = f"{seed_prefix}-{floorplan_id}-conj-sup-{idx:06d}"
        rows.append(
            make_record(
                rec_id=rid,
                claim=claim,
                label="SUPPORTED",
                structural_reasoning="conjunction",
                evidence_type="Perception",
                evidence_triples=ev,
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes=None,
            )
        )

    # Refuted conjunctions: corrupt one part (ensure actually refuted)
    for idx in range(n_refuted):
        if len(candidates) < 2:
            break

        a, b = random.sample(candidates, 2)  # avoids duplicates

        a_t = tuple(a["evidence"]["triples"][0])
        b_t = tuple(b["evidence"]["triples"][0])

        # try a few times to get a genuinely false corruption
        b_bad = None
        for _ in range(10):
            trial = corrupt_triple(b_t, triples, triple_set)
            if trial not in triple_set and trial != b_t:
                b_bad = trial
                break
        if b_bad is None:
            continue  # couldn't corrupt reliably; skip

        bad_claim = verbalize(b_bad)
        if bad_claim is None or bad_claim.strip() == "":
            continue

        claim = make_conjunction_claim(a["claim"], bad_claim)
        ev = [a_t, b_bad]

        rid = f"{seed_prefix}-{floorplan_id}-conj-ref-{idx:06d}"
        rows.append(
            make_record(
                rec_id=rid,
                claim=claim,
                label="REFUTED",  # guaranteed by construction above
                structural_reasoning="conjunction",
                evidence_type="Perception",
                evidence_triples=ev,
                evidence_source="AI2-THOR-RDF",
                evidence_source_type="perception",
                evidence_urls=[],
                scene_id=floorplan_id,
                generator="ai2thor",
                split=None,
                notes="corrupted_conjunction",
            )
        )


    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", type=Path, required=True, help="Path to a Turtle (.ttl) file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_floorplans", type=int, default=0, help="0 = all detected floorplans")
    parser.add_argument("--onehop_per_floorplan", type=int, default=200)
    parser.add_argument("--neg_pairs_per_floorplan", type=int, default=60)
    parser.add_argument("--conj_per_floorplan", type=int, default=80)
    args = parser.parse_args()

    random.seed(args.seed)
    seed_prefix = f"rdf{args.seed}"

    load = load_ttl_triples(args.ttl)
    grouped = group_by_floorplan(load.triples)

    floorplans = sorted(grouped.keys())
    if args.max_floorplans and args.max_floorplans > 0:
        floorplans = floorplans[: args.max_floorplans]

    all_rows: List[Dict[str, Any]] = []

    print(floorplans)

    for fp in floorplans:
        triples = grouped[fp]

        # Only keep predicates we can verbalize
        usable_triples = [t for t in triples if is_verbalizable_predicate(t[1])]
        usable_set = set(usable_triples)

        if not usable_triples:
            continue

        onehop_rows = generate_one_hop(
            floorplan_id=fp,
            triples=usable_triples,        # ✅ use filtered
            triple_set=usable_set,         # ✅ use filtered
            n_supported=args.onehop_per_floorplan // 2,
            n_refuted=args.onehop_per_floorplan // 2,
            seed_prefix=seed_prefix,
        )
        all_rows.extend(onehop_rows)

        neg_rows = generate_negation_pairs(
            floorplan_id=fp,
            triples=usable_triples,
            triple_set=usable_set,
            max_pairs=args.neg_pairs_per_floorplan,
            seed_prefix=seed_prefix,
        )
        all_rows.extend(neg_rows)

        # Conjunction needs supported one-hop rows
        supported_onehop = [r for r in onehop_rows if r.get("label") == "SUPPORTED"]
        conj_rows = generate_conjunction(
            floorplan_id=fp,
            supported_onehop=supported_onehop,
            triples=usable_triples,        # ✅ use filtered
            triple_set=usable_set,         # ✅ use filtered
            n_supported=args.conj_per_floorplan // 2,
            n_refuted=args.conj_per_floorplan // 2,
            seed_prefix=seed_prefix,
        )
        all_rows.extend(conj_rows)

    random.shuffle(all_rows)

    # Create output folder
    folder = args.output.parent
    if folder and not folder.exists():
        os.makedirs(folder, exist_ok=True)
    write_jsonl(args.output, all_rows)
    print(f"Wrote {len(all_rows)} records to {args.output}")


if __name__ == "__main__":
    main()
