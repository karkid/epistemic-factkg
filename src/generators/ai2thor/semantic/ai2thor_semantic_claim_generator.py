from __future__ import annotations

from email.utils import unquote
from knowledge_graph.ontology.base import BaseOntology
from knowledge_graph.semantics.claims.base import BaseClaimGenerator

from utils.typing import Triple


class AI2ThorSemanticClaimGenerator(BaseClaimGenerator):
    """
    Generate textual claims specific to AI2Thor semantic data.
    """

    def __init__(self, *, ontology: BaseOntology):
        super().__init__()
        self.ontology = ontology

    def _short(self, x: str) -> str:
        """
        Make URIs readable:
        - decode URL encoding (%7C -> |)
        - take last fragment after # or /
        """
        x = unquote(x)  # ⭐ decode first

        if "#" in x:
            x = x.split("#")[-1]
        if "/" in x:
            x = x.split("/")[-1]

        return x

    def _pretty_objtype(self, x: str) -> str:
        # Mug|... -> Mug, Table_3 -> Table, etc.
        x = self._short(x)
        x = x.split("|")[0]
        x = x.split("_")[0]  # simple heuristic
        return x

    def _is_type_predicate(self, pred: str) -> bool:
        """
        Detect rdf:type or any predicate ending with 'type'.
        We must exclude these because they create nonsense claims.
        """
        ps = pred.split("#")[-1].split("/")[-1].lower()
        return ps == "type"

    def is_verbalizable_predicate(self, pred: str) -> bool:
        ps = pred.split("#")[-1].split("/")[-1].lower()
        if ps in {"type", "rdf:type", "inscene", "hasobject"}:
            return False
        return ps in {"ontopof", "inside", "isopen", "istoggled", "isdirty", "isbroken"}

    def verbalize(self, triple: Triple) -> str:
        s, p, o = triple
        subj = self._pretty_objtype(s)
        pred = self._short(p)
        obj = self._pretty_objtype(o)

        pred_l = pred.lower()

        if pred_l.endswith("ontopof") or pred_l == "ontopof":
            return f"A {subj.lower()} is on the {obj.lower()}."
        if pred_l.endswith("inside") or pred_l == "inside":
            return f"A {subj.lower()} is inside the {obj.lower()}."

        if pred_l.endswith("isopen") or pred_l == "isopen":
            val = str(o).lower()
            return (
                f"The {subj.lower()} is open."
                if val == "true"
                else f"The {subj.lower()} is not open."
            )

        if pred_l.endswith("istoggled") or pred_l == "istoggled":
            val = str(o).lower()
            return (
                f"The {subj.lower()} is switched on."
                if val == "true"
                else f"The {subj.lower()} is not switched on."
            )

        if pred_l.endswith("isdirty") or pred_l == "isdirty":
            val = str(o).lower()
            return (
                f"The {subj.lower()} is dirty."
                if val == "true"
                else f"The {subj.lower()} is not dirty."
            )

        if pred_l.endswith("isbroken") or pred_l == "isbroken":
            val = str(o).lower()
            return (
                f"The {subj.lower()} is broken."
                if val == "true"
                else f"The {subj.lower()} is not broken."
            )

        if pred_l.endswith("hasobject") or pred_l == "hasobject":
            return ""
        elif self._is_type_predicate(p):
            return ""

        return ""
