from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetValidator(ABC):
    """
    Port interface for dataset-specific validation rules applied on top of
    the JSON Schema check in validate_unified_dataset.py.

    Implement one subclass per dataset and register it in the validator
    registry so that validate_unified_dataset.py can dispatch automatically.

    Example
    -------
    class AI2ThorValidator(DatasetValidator):
        @property
        def dataset_name(self) -> str:
            return "ai2thor"

        def check(self, record: dict) -> list[str]:
            msgs = []
            if record.get("claim_triples") is None:
                msgs.append("AI2THOR record missing claim_triples.")
            return msgs
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Must match the value written into provenance.dataset by the converter.
        Example: 'averitec', 'ai2thor'.
        """

    @abstractmethod
    def check(self, record: dict) -> list[str]:
        """
        Run dataset-specific consistency checks on a unified v2.0 record.

        Parameters
        ----------
        record : dict — a single unified v2.0 record (already schema-validated)

        Returns
        -------
        list[str] — warning/error messages. Empty list means no issues.
        """
