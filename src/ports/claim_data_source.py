# src/ports/claim_data_source.py
from abc import ABC, abstractmethod
from typing import Iterator, List
from src.core.claims.types import ClaimInstance

class ClaimDataSource(ABC):
    
    @abstractmethod
    def get_claims(self) -> Iterator[ClaimInstance]:
        """Yield claim instances one by one."""
        raise NotImplementedError

    @abstractmethod
    def get_claim_by_id(self, claim_id: str) -> ClaimInstance:
        """Return a single claim instance by its ID."""
        raise NotImplementedError

    @abstractmethod
    def get_available_claims(self) -> List[str]:
        """Return a list of all claim IDs available in this datasource."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """
        Optional cleanup hook.

        Most datasources won't need it, but DB connections / file handles might.
        """
        return
