"""
Custom exceptions for the epistemic-factkg project.

This module defines a hierarchy of exceptions that provide better error handling
and debugging capabilities throughout the system.
"""


class FactKGError(Exception):
    """Base exception for all epistemic-factkg related errors."""

    pass


class DataSourceError(FactKGError):
    """Base exception for data source operations."""

    pass

class DataLoadError(DataSourceError):
    """Errors during data loading operations (TTL parsing, scene loading, etc.)."""

    pass


class ValidationError(DataSourceError):
    """Data validation errors (malformed objects, missing required fields)."""

    pass

class ConfigurationError(FactKGError):
    """Configuration-related errors (missing files, invalid config, etc.)."""

    pass

class OntologyError(FactKGError):
    """Ontology and predicate mapping related errors."""

    pass


class BuildError(FactKGError):
    """Knowledge graph building errors."""

    pass


class SemanticError(FactKGError):
    """Semantic claim processing errors."""

    pass


class ClaimValidationError(SemanticError):
    """Errors during claim validation and processing."""

    pass
