"""Epistemic framework — single source of truth for fact-verification logic.

This package is the foundation for all other packages. It contains:
- Verdict, EvidenceStance, EvidenceType, ReasoningStrategy enums
- EC formula (epistemic confidence computation)
- Source trust registry resolution
- Schema definitions and validation

Nothing in src/epistemic/ imports from other src/* packages.
Everything else (adapters, model, pipeline) imports from here.
"""

__all__ = []
