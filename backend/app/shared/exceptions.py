"""Shared exception hierarchy used across layers."""


class ParsingError(Exception):
    """Raised when pseudocode parsing fails."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}
