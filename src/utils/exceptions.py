"""
Custom exceptions for the epistemic-factkg project.

This module defines a hierarchy of exceptions that provide better error handling
and debugging capabilities throughout the system.
"""


class EpistemicFactKGError(Exception):
    """Base exception for all epistemic-factkg related errors."""

    pass


class DataSourceError(EpistemicFactKGError):
    """Base exception for data source operations."""

    pass


class ConfigurationError(DataSourceError):
    """Configuration-related errors (missing files, invalid config, etc.)."""

    pass


class DataLoadError(DataSourceError):
    """Errors during data loading operations (TTL parsing, scene loading, etc.)."""

    pass


class ValidationError(DataSourceError):
    """Data validation errors (malformed objects, missing required fields)."""

    pass


class OntologyError(EpistemicFactKGError):
    """Ontology and predicate mapping related errors."""

    pass


class BuildError(EpistemicFactKGError):
    """Knowledge graph building errors."""

    pass


class SemanticError(EpistemicFactKGError):
    """Semantic claim processing errors."""

    pass


class ClaimValidationError(SemanticError):
    """Errors during claim validation and processing."""

    pass
