"""EpistemicPredictor — inference backend for the Streamlit demo app."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import torch

from src.epistemic.formula import compute_evidence_confidence
from src.epistemic.registry import (
    get_source_trust,
    load_source_trust_registry,
    resolve_source_id,
)
from src.model.config import GraphConfig
from src.model.data.builder import ClaimGraphBuilder
from src.model.data.featurizer import Featurizer
from src.model.data.types import VERDICT_TO_INT, NodeType
from src.model.models import MODELS
from src.model.models.nlihybridhgnn import NLIHybridHGNN

_INT_TO_VERDICT = {v: k for k, v in VERDICT_TO_INT.items()}
_INT_TO_STANCE = {0: "supports", 1: "refutes", 2: "neutral"}

_SOURCE_ID_TYPE_MAP: list[tuple[str, str]] = [
    ("academic", "academic"),
    ("wikipedia", "academic"),
    ("scholar", "academic"),
    ("pubmed", "academic"),
    ("arxiv", "academic"),
    ("news", "news"),
    ("reuters", "news"),
    ("bbc", "news"),
    ("cnn", "news"),
    ("guardian", "news"),
    ("government", "government"),
    ("_gov", "government"),
    ("social", "social_media"),
    ("twitter", "social_media"),
    ("reddit", "social_media"),
    ("ai2thor", "simulation"),
    ("simulation", "simulation"),
]

_ARCHIVE_RE = re.compile(r"web\.archive\.org/web/\d+[^/]*/(.+)")

_SOURCE_TYPE_DEFAULTS: dict[str, tuple[str, float]] = {
    "news": ("general_web_text", 0.60),
    "academic": ("academic_pdf", 0.85),
    "government": ("government_web_text", 0.85),
    "social_media": ("social_media_web_text", 0.35),
    "simulation": ("ai2thor_simulation", 1.0),
    "unknown": ("unknown_web", 0.40),
}

_PRAMANA_LABEL: dict[str, str] = {
    "web_text": "Testimony (Shabda)",
    "pdf": "Testimony (Shabda)",
    "image": "Perception (Pratyaksha)",
    "video": "Perception (Pratyaksha)",
    "audio": "Perception (Pratyaksha)",
    "web_table": "Comparison (Upamana)",
}


def _infer_types(modality: str) -> list[str]:
    if modality in ("image", "video", "audio"):
        return ["perception"]
    if modality == "web_table":
        return ["comparison_analogy", "testimony"]
    return ["testimony"]


def _pramana_label(modality: str) -> str:
    return _PRAMANA_LABEL.get(modality, "Testimony (Shabda)")


def _is_from_trust(trust: float) -> float:
    return round(min(0.8, max(0.1, trust)), 4)


class EpistemicPredictor:
    """Loads a trained checkpoint and runs inference on free-text claim + evidence."""

    def __init__(self, model_name: str = "v2-hgnn") -> None:
        self.model_name = model_name
        checkpoint = Path(f"out/model/{model_name}/checkpoints/best_model.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint}. Run: just train {model_name}"
            )

        self._registry = load_source_trust_registry(
            "data/registry/source_trust_registry.jsonl"
        )
        is_nli = MODELS.get(model_name) is NLIHybridHGNN
        self._builder = ClaimGraphBuilder(
            registry=self._registry,
            featurizer=Featurizer(),
            use_nli=is_nli,
        )

        model_cls = MODELS[model_name]
        graph_cfg = GraphConfig.v2() if is_nli else GraphConfig.v1()
        self._model = model_cls(graph_cfg, hidden_dim=256, heads=4, dropout=0.3)
        self._model.load_state_dict(
            torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, claim: str, evidence_items: list[dict]) -> dict:
        """Run full neuro-symbolic inference on a claim + evidence list.

        Args:
            claim: Free-text claim string.
            evidence_items: List of dicts, each with keys:
                text (str), url (str|None), source_type (str), modality (str).

        Returns dict with keys:
            verdict, verdict_probs, support_score, refute_score,
            has_ec, evidence_breakdown.
        """
        resolved = self._resolve_sources(evidence_items)
        record = self._build_record(claim, resolved)
        graph = self._builder.build(record)

        if graph is None:
            raise ValueError(
                "All evidence items were filtered (boilerplate, empty, or duplicate). "
                "Please provide at least one non-empty evidence text."
            )

        with torch.no_grad():
            out = self._model.predict(graph.data)

        stance_pred = out["stance_pred"]
        stance_logits = out["stance_logits"]
        is_pred = out["is_pred"]

        stance_probs = torch.softmax(stance_logits, dim=-1)
        verdict_probs = torch.softmax(out["verdict_logits"], dim=-1)[0].tolist()

        ev_data = graph.data[NodeType.EVIDENCE]
        ew_vals = ev_data.ew.tolist()
        st_vals = ev_data.st.tolist()
        is_vals = is_pred.view(-1).tolist()

        is_nli = (self.model_name == "v3-nli")
        nli_raw = ev_data.x[:, -3:].tolist() if is_nli else []

        n_real = len(evidence_items)
        breakdown = []
        for i in range(min(n_real, len(stance_pred))):
            s_idx = int(stance_pred[i].item())
            st = st_vals[i]
            ew = ew_vals[i]
            is_val = is_vals[i]
            ec = compute_evidence_confidence(st, ew, is_val)
            ev = evidence_items[i]
            text = ev["text"]
            item: dict = {
                "text": text,
                "text_short": (text[:150] + "…") if len(text) > 150 else text,
                "pramana": _pramana_label(ev.get("modality", "web_text")),
                "modality": ev.get("modality", "web_text"),
                "source_type": ev.get("source_type", "unknown"),
                "stance": _INT_TO_STANCE[s_idx],
                "stance_confidence": round(float(stance_probs[i, s_idx].item()), 3),
                "is_score": round(is_val, 3),
                "source_trust": round(st, 3),
                "evidence_weight": round(ew, 3),
                "ec_score": round(ec, 3),
                "source_id": resolved[i]["source_id"],
                "nli_probs": None,
            }
            if is_nli and i < len(nli_raw):
                raw = nli_raw[i]
                item["nli_probs"] = {
                    "contradiction": round(raw[0], 3),
                    "entailment": round(raw[1], 3),
                    "neutral": round(raw[2], 3),
                }
            breakdown.append(item)

        return {
            "verdict": out["verdict"],
            "verdict_probs": verdict_probs,
            "support_score": float(out.get("support_score", 0.0)),
            "refute_score": float(out.get("refute_score", 0.0)),
            "has_ec": self.model_name != "baseline",
            "evidence_breakdown": breakdown,
            "hetero_data": graph.data,  # HeteroData — used for pyvis graph viz
        }

    def predict_from_record(self, record: dict) -> dict:
        """Run prediction on a raw training/test record dict."""
        raw_ev = record.get("evidence", [])
        evidence_items = [
            {
                "text": ev["text"],
                "source_type": self._source_id_to_type(ev.get("source_id", "")),
                "modality": ev.get("modality", "web_text"),
            }
            for ev in raw_ev
        ]
        result = self.predict(record["claim"], evidence_items)
        for i, bd in enumerate(result["evidence_breakdown"]):
            if i < len(raw_ev):
                bd["triples"] = raw_ev[i].get("triples") or []
        result["claim_triples"] = record.get("claim_triples") or []
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_sources(self, evidence_items: list[dict]) -> list[dict]:
        result = []
        for ev in evidence_items:
            source_id, trust = self._resolve_source(
                ev.get("url"), ev.get("source_type", "unknown")
            )
            result.append({**ev, "source_id": source_id, "resolved_trust": trust})
        return result

    def _resolve_source(self, url: str | None, source_type: str) -> tuple[str, float]:
        if url:
            url = url.strip()
            # Extract original URL from Wayback Machine archives (ADR-020)
            m = _ARCHIVE_RE.search(url)
            if m:
                embedded = m.group(1)
                if not embedded.startswith(("http://", "https://")):
                    embedded = "https://" + embedded
                url = embedded
            parsed = urlparse(url if "://" in url else "https://" + url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            domain = domain.lower().removeprefix("www.")
            if domain:
                modality = "pdf" if url.lower().endswith(".pdf") else "web_text"
                source_id = resolve_source_id(domain, modality, self._registry)
                trust = get_source_trust(source_id, self._registry)
                return source_id, trust
        source_id, trust = _SOURCE_TYPE_DEFAULTS.get(
            source_type, ("unknown_web", 0.40)
        )
        return source_id, trust

    def _source_id_to_type(self, source_id: str) -> str:
        sid = source_id.lower()
        for pattern, stype in _SOURCE_ID_TYPE_MAP:
            if pattern in sid:
                return stype
        return "unknown"

    def _build_record(self, claim: str, resolved_items: list[dict]) -> dict:
        evidence = []
        for ev in resolved_items:
            modality = ev.get("modality", "web_text")
            trust = ev.get("resolved_trust", 0.40)
            evidence.append(
                {
                    "text": ev["text"],
                    "stance": None,
                    "modality": modality,
                    "evidence_types": _infer_types(modality),
                    "inference_strength": _is_from_trust(trust),
                    "source_id": ev["source_id"],
                    "triples": [],
                }
            )
        return {
            "claim": claim,
            "evidence": evidence,
            "reasoning": {"strategy": "testimonial_lookup"},
            "verdict": {"label": "not_enough_evidence"},
            "provenance": {"dataset": "demo"},
        }
