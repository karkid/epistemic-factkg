from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

from src.core.graph.types import Graph


class GraphDataSource(ABC):
    """
    Port/interface for anything that can yield Graph objects.

    Implementations live in adapters/, e.g.:
    - adapters/ai2thor/graph_data_source.py
    - adapters/jsonl/graph_data_source.py
    - adapters/db/graph_data_source.py
    """

    @abstractmethod
    def get_graphs(self) -> Iterator[Graph]:
        """Yield graphs one by one."""
        raise NotImplementedError

    @abstractmethod
    def get_graph_by_id(self, graph_id: str) -> Graph:
        """Return a single graph by its ID."""
        raise NotImplementedError

    @abstractmethod
    def get_available_graph_ids(self) -> List[str]:
        """Return a list of all graph IDs available in this datasource."""
        raise NotImplementedError

    def has_graph(self, graph_id: str) -> bool:
        """Convenience method (optional)."""
        return graph_id in set(self.get_available_graph_ids())

    def cleanup(self) -> None:
        """
        Optional cleanup hook.

        Most datasources won't need it, but DB connections / file handles might.
        """
        return
