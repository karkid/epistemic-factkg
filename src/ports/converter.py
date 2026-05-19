from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class DatasetConverter(ABC):
    """
    Port interface for converting any source dataset into unified v3.0 JSONL.

    Implement one subclass per dataset (AVeriTeC, FEVER, etc.) or simulation
    source (AI2THOR, Habitat, etc.) and place it in src/adapters/{source_name}/.

    Example
    -------
    class AveritecConverter(DatasetConverter):
        @property
        def dataset_name(self) -> str:
            return "averitec"

        def convert_one(self, raw: dict, rec_id: str) -> dict: ...
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Short lowercase identifier written into provenance.dataset.
        Use only [a-z0-9_] characters. Example: 'averitec', 'ai2thor', 'fever'.
        """

    @abstractmethod
    def convert_one(self, raw_record: dict, rec_id: str) -> dict:
        """
        Convert a single raw source record to a unified v3.0 dict.

        Parameters
        ----------
        raw_record : dict
            One record as loaded from the source file (JSON/JSONL).
        rec_id : str
            Stable identifier to assign as the record's 'id' field if the
            source does not supply one.

        Returns
        -------
        dict
            A dict that passes validation against src/epistemic/schema.py:CLAIM_SCHEMA.
        """

    def iter_records(self, in_path: str) -> Iterator[tuple[int, dict]]:
        """
        Yield (index, raw_record) from a source file.
        Default: reads JSONL. Override for JSON arrays or other formats.
        """
        import json

        with open(in_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    yield i, json.loads(line)

    def convert_file(
        self, in_path: str, out_path: str, split: str | None = None
    ) -> int:
        """
        Convert an entire source file to unified v3.0 JSONL.

        Each converted record is validated against CLAIM_SCHEMA before being
        written. Records that fail schema validation are skipped and logged so
        they are excluded from all downstream steps rather than silently
        corrupting the dataset.

        Parameters
        ----------
        in_path  : str — source file path
        out_path : str — destination JSONL path
        split    : str | None — 'train' | 'dev' | 'test' | None

        Returns
        -------
        int — number of valid records written (excludes skipped records)
        """
        import json
        from jsonschema import Draft7Validator
        from src.epistemic.schema import CLAIM_SCHEMA

        validator = Draft7Validator(CLAIM_SCHEMA)
        count = 0
        skipped = 0

        with open(out_path, "w", encoding="utf-8") as fout:
            for i, raw in self.iter_records(in_path):
                rec_id = f"{self.dataset_name}-{split or 'unknown'}-{i:06d}"
                unified = self.convert_one(raw, rec_id)
                if split is not None and "provenance" in unified:
                    unified["provenance"]["split"] = split

                errors = list(validator.iter_errors(unified))
                if errors:
                    record_id = unified.get("id", rec_id)
                    for err in errors:
                        field = ".".join(str(p) for p in err.absolute_path) or "(root)"
                        print(
                            f"  [SKIP] {record_id}: schema error at {field!r}: {err.message}",
                            flush=True,
                        )
                    skipped += 1
                    continue

                fout.write(json.dumps(unified, ensure_ascii=False) + "\n")
                count += 1

        if skipped:
            print(
                f"  [{self.dataset_name}] {count} records written, "
                f"{skipped} skipped (schema invalid) — fix the converter or source data.",
                flush=True,
            )
        return count
