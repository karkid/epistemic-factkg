import json
import argparse
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# -----------------------
# Utility
# -----------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_entity_uri(uri: str) -> Tuple[str, Optional[Tuple[float, float, float]]]:
    """
    AI2-THOR URIs look like:
    http://.../entities/Cabinet%7C-01.46%7C%2B00.78%7C%2B00.47
    Decode object type and coordinates if present.
    """
    if not isinstance(uri, str):
        return ("Unknown", None)
    tail = uri.split("/")[-1]
    tail = tail.replace("%7C", "|")
    tail = tail.replace("%2B", "+")
    parts = tail.split("|")
    obj_type = parts[0] if parts else "Unknown"
    coords = None
    if len(parts) >= 4:
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coords = (x, y, z)
        except Exception:
            coords = None
    return obj_type, coords

def rel_name(uri: str) -> str:
    # relation URI -> short name
    if not isinstance(uri, str):
        return "unknown_relation"
    return uri.split("/")[-1]

def normalize_label(lbl: Optional[str]) -> Optional[str]:
    if lbl is None:
        return None
    lbl = lbl.strip().lower()
    mapping = {"support": "supported", "supported": "supported",
               "refute": "refuted", "refuted": "refuted"}
    return mapping.get(lbl, lbl)

def stance_from_label(lbl: Optional[str]) -> str:
    if lbl == "supported":
        return "supports"
    if lbl == "refuted":
        return "refutes"
    return "unknown"

# -----------------------
# Claim type + strategy rules (your tables)
# -----------------------
def classify_claim_type(predicate: str, obj_value: str) -> str:
    """
    Returns one of: Spatial, State, Property, Perceptual, Affordance
    """
    pred = predicate.lower()

    # Spatial
    if pred in {"inside", "ontopof", "near", "in", "on"}:
        return "Spatial"
    if pred in {"onTopOf".lower(), "inside".lower(), "near".lower()}:
        return "Spatial"

    # State / Perceptual / Property / Affordance
    if pred == "temperature":
        return "Perceptual"
    if pred in {"dirty", "broken"}:
        return "State"
    if pred in {"isopen", "open", "isclosed", "closed"}:
        return "State"
    if pred in {"material", "color"}:
        return "Property"
    if pred in {"breakable", "pickupable", "openable"}:
        return "Affordance"

    # fallback: if object value looks like a temperature/state token
    if obj_value.lower() in {"roomtemp", "hot", "cold"}:
        return "Perceptual"

    return "Property"

def classify_strategy(predicate: str, obj_value: str) -> str:
    """
    Returns one of: perception, spatial_reasoning, action_testing, simulation_metadata
    """
    pred = predicate.lower()
    val = (obj_value or "").lower()

    # perception-like
    if pred == "temperature" or val in {"roomtemp", "hot", "cold"}:
        return "perception"
    if pred in {"dirty", "broken"}:
        return "perception"
    if pred in {"isopen", "open", "isclosed", "closed"}:
        return "perception"

    # spatial relations
    if pred in {"inside", "ontopof", "near"}:
        return "spatial_reasoning"

    # affordances
    if pred in {"breakable", "pickupable", "openable"}:
        return "action_testing"

    # material/color
    if pred in {"material", "color"}:
        return "simulation_metadata"

    return "perception"

# -----------------------
# Text templates
# -----------------------
def make_justification(label: str, predicate: str, claim_triples: List[List[str]], evidence_triples: List[List[str]]) -> str:
    """
    Generate a short justification using predicate + label.
    """
    pred = predicate.lower()

    if pred == "temperature":
        # evidence_triples typically include the observed value
        observed = evidence_triples[0][2] if evidence_triples and len(evidence_triples[0]) == 3 else "unknown"
        claimed = claim_triples[0][2] if claim_triples and len(claim_triples[0]) == 3 else "unknown"
        if label == "supported":
            return f"The simulator observation shows temperature = {observed}, matching the claim."
        else:
            return f"The simulator observation shows temperature = {observed}, which contradicts the claimed value ({claimed})."

    if pred in {"ontopof", "inside", "near"}:
        # spatial relation
        if label == "supported":
            return f"The simulator observation confirms the stated spatial relation ({predicate})."
        else:
            return f"The simulator observation contradicts the stated spatial relation ({predicate})."

    # generic fallback
    if label == "supported":
        return "The simulator observation matches the claim's stated relation/value."
    return "The simulator observation contradicts the claim's stated relation/value."

def make_proof_rationale(evidence_source: str, evidence_source_type: str, predicate: str) -> str:
    pred = predicate.lower()
    if evidence_source == "simulation" and evidence_source_type == "perception":
        if pred in {"ontopof", "inside", "near"}:
            return "Evidence comes from simulator perception and is checked via spatial state relations."
        return "Evidence comes from simulator perception with direct state verification."
    return "Evidence derived from simulator metadata."

def make_proof_confidence(evidence_source: str, evidence_source_type: str, predicate: str) -> float:
    pred = predicate.lower()
    if evidence_source == "simulation" and evidence_source_type == "perception":
        # You can tune this; simple heuristic:
        if pred in {"ontopof", "inside", "near"}:
            return 0.95
        return 1.0
    return 0.8

# -----------------------
# Main conversion
# -----------------------
def convert_one(ai2: Dict[str, Any], split: Optional[str] = None) -> Dict[str, Any]:
    oid = ai2.get("id")
    claim_text = ai2.get("claim")
    label = normalize_label(ai2.get("label"))

    claim_triples = ai2.get("claim_triples") or []
    reasoning = ai2.get("reasoning") or {}
    evidence = ai2.get("evidence") or {}
    context = ai2.get("context") or {}
    meta = ai2.get("meta") or {}

    evidence_triples = evidence.get("evidence_triples") or []
    evidence_source = (evidence.get("evidence_source") or "simulation").lower()
    evidence_source_type = (evidence.get("evidence_source_type") or "perception").lower()
    evidence_urls = evidence.get("evidence_urls") or []

    # derive predicate and object value from first claim triple (works for one-hop/conjunction too)
    first_triple = claim_triples[0] if claim_triples else ["", "", ""]
    predicate = rel_name(first_triple[1]) if len(first_triple) == 3 else "unknown_relation"
    obj_value = first_triple[2] if len(first_triple) == 3 else ""

    # auto fields requested
    claim_type = classify_claim_type(predicate, obj_value)
    strategy = classify_strategy(predicate, obj_value)

    justification = make_justification(label, predicate, claim_triples, evidence_triples)
    proof_rationale = make_proof_rationale(evidence_source, evidence_source_type, predicate)
    proof_conf = make_proof_confidence(evidence_source, evidence_source_type, predicate)

    created_utc = meta.get("created_utc") or now_utc_iso()
    context_id = context.get("context_id")
    source_url = evidence_urls[0] if evidence_urls else (f"ai2thor://scene/{context_id}" if context_id else None)

    out = {
        "id": oid,
        "claim": claim_text,
        "verdict": {
            "label": label,
            "justification": justification,
            "annotator_confidence": None
        },
        "epistemic": {
            "proof_types": ["perception"],          # multi-label (list)
            "primary_proof_type": "perception",     # single label for training/plots
            "proof_type_rationale": proof_rationale,
            "proof_confidence": proof_conf
        },
        "claim_meta": {
            "required_reannotation": False,
            "claim_date": created_utc[:10] if isinstance(created_utc, str) and len(created_utc) >= 10 else None,
            "speaker": "automation_agent",
            "original_claim_url": None,
            "cached_original_claim_url": None,
            "fact_checking_article": None,
            "reporting_source": "ai2thor_simulation",
            "location_ISO_code": "VIRTUAL",
            "claim_types": [claim_type],
            "fact_checking_strategies": [strategy],
            "claim_id": None
        },
        "claim_triples": claim_triples if claim_triples else None,
        "reasoning": {
            "structural": reasoning.get("structural"),
            "type": "direct_observation" if evidence_source_type == "perception" else None
        } if reasoning.get("structural") else None,
        "qa": None,
        "evidence_items": [
            {
                "evidence_id": f"{oid}-e0",
                "source_type": "simulation",
                "source_url": source_url,
                "cached_source_url": None,
                "source_medium": "simulation_state",
                "stance": stance_from_label(label),
                "extract": None,
                "evidence_triples": evidence_triples if evidence_triples else None
            }
        ],
        "context": {
            "context_id": context_id,
            "context_type": context.get("context_type"),
            "generator": context.get("generator"),
            "split": split if split is not None else context.get("split")
        },
        "meta": {
            "created_utc": created_utc,
            "notes": meta.get("notes")
        }
    }
    return out

def convert_ai2thor_file(infile: str, outfile: str, split: str = None):
    data = read_jsonl(infile)
    unified = [convert_one(x, split=split) for x in data]
    write_jsonl(outfile, unified)
