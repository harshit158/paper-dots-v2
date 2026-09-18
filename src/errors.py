"""Application-level errors shared across layers."""


class ValidationError(ValueError):
    """Raised when user input or an external response is invalid."""
