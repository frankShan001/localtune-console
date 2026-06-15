"""Stable API errors shared by services and Flask routes."""

from __future__ import annotations


class LocalTuneError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def error_details(error: Exception, default_code: str = "REQUEST_FAILED") -> tuple[str, str, int]:
    if isinstance(error, LocalTuneError):
        return error.code, str(error), error.status
    if isinstance(error, (ValueError, TypeError)):
        return "VALIDATION_ERROR", str(error), 400
    return default_code, str(error), 500
