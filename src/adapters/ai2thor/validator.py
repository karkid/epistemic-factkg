from __future__ import annotations

from src.core.ports.dataset.validator import DatasetValidator


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

        if stance == "absent" and ev_triples:
            msgs.append(
                "AI2THOR absence claim (stance=absent) has non-empty evidence triples."
            )

        if pramana == "non_apprehension" and stance != "absent":
            msgs.append(
                f"AI2THOR non_apprehension claim expected stance=absent, got {stance!r}."
            )

        if record.get("claim_triples") is None:
            msgs.append("AI2THOR record missing claim_triples.")

        if record.get("reasoning") is None:
            msgs.append("AI2THOR record missing reasoning block.")

        if pramana not in ("perception", "non_apprehension"):
            msgs.append(f"AI2THOR record has unexpected pramana_primary: {pramana!r}.")

        return msgs
