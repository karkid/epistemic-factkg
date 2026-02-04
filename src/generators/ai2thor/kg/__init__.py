"""
AI2-THOR knowledge graph integration package.
"""

from .data_source import AI2THORDataSource
from .ontology import create_ai2thor_ontology

__all__ = ["AI2THORDataSource", "create_ai2thor_ontology"]
