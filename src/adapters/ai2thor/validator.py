from __future__ import annotations

from src.core.ports.dataset.validator import DatasetValidator
from src.core.claims.labels import EvidenceStance, Pramana


# Valid Pramana types for AI2THOR records — only perception-based sources
# are valid since the simulator provides direct/complete world state.
_VALID_AI2THOR_PRAMANAS = {Pramana.PERCEPTION.value, Pramana.NON_APPREHENSION.value}


class AI2ThorValidator(DatasetValidator):
    """AI2THOR-specific consistency checks on unified v2.0 records."""

    @property
    def dataset_name(self) -> str:
        return "ai2thor"

    def check(self, record: dict) -> list[str]:
        msgs = []
        evidence = record.get("evidence") or []

        if not evidence:
            msgs.append("AI2THOR record has no evidence items.")
            return msgs

        ev = evidence[0]
        stance = ev.get("stance")
        ev_triples = ev.get("triples") or []
        pramana = (record.get("epistemic") or {}).get("pramana_primary")

        # Absence claims must have no evidence triples — the missing object IS the evidence
        if stance == EvidenceStance.ABSENT.value and ev_triples:
            msgs.append(
                "AI2THOR absence claim (stance=absent) has non-empty evidence triples."
            )

        # non_apprehension pramana must always pair with stance=absent
        if (
            pramana == Pramana.NON_APPREHENSION.value
            and stance != EvidenceStance.ABSENT.value
        ):
            msgs.append(
                f"AI2THOR non_apprehension claim expected stance=absent, got {stance!r}."
            )

        structural = (record.get("reasoning") or {}).get("structural")
        # Absence claims (non_apprehension or structural=absence) legitimately have
        # no claim_triples — the absent object cannot be represented as a positive triple.
        if (
            pramana != Pramana.NON_APPREHENSION.value
            and structural != "absence"
            and record.get("claim_triples") is None
        ):
            msgs.append("AI2THOR record missing claim_triples.")

        if record.get("reasoning") is None:
            msgs.append("AI2THOR record missing reasoning block.")

        # Only perception / non_apprehension are valid for simulator-sourced records
        if pramana not in _VALID_AI2THOR_PRAMANAS:
            msgs.append(f"AI2THOR record has unexpected pramana_primary: {pramana!r}.")

        return msgs
