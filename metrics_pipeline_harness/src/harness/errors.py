"""Structured errors and CLI exit codes (plan §12.2).

Every error that can leave the harness carries a stable machine-readable ``code`` (for example
``WB_ZIP_LIMIT``) and maps to one of the documented exit codes.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    OK = 0
    INTERNAL = 1
    USAGE = 2
    VALIDATION_FAILED = 3
    COMPLETED_WITH_BLOCKED = 4
    GATE_DENIED = 5
    TESTS_FAILED = 6
    RESOURCE_LIMIT = 7
    STALE_PREREQUISITE = 8
    TRANSIENT_TARGET = 9
    CANCELLED = 130


class HarnessError(Exception):
    """Base class. ``code`` is stable and appears in issues, logs and reports."""

    exit_code: ExitCode = ExitCode.INTERNAL

    def __init__(self, code: str, message: str = "", *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.evidence = evidence or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
        }


class ConfigError(HarnessError):
    exit_code = ExitCode.USAGE


class ValidationFailed(HarnessError):
    exit_code = ExitCode.VALIDATION_FAILED


class ResourceLimitExceeded(HarnessError):
    exit_code = ExitCode.RESOURCE_LIMIT


class UnsupportedWorkbook(HarnessError):
    """The file is not a format the current ingestion path accepts (routed, never guessed)."""

    exit_code = ExitCode.VALIDATION_FAILED


class WorkbookCorrupt(HarnessError):
    exit_code = ExitCode.VALIDATION_FAILED


class ArtifactWriteFailed(HarnessError):
    exit_code = ExitCode.INTERNAL


ERROR_TYPES: dict[str, type[HarnessError]] = {
    cls.__name__: cls
    for cls in (
        HarnessError,
        ConfigError,
        ValidationFailed,
        ResourceLimitExceeded,
        UnsupportedWorkbook,
        WorkbookCorrupt,
        ArtifactWriteFailed,
    )
}


def error_from_dict(data: dict[str, Any]) -> HarnessError:
    """Rebuild an error serialized by :meth:`HarnessError.as_dict` (used across the bounded subprocess)."""
    cls = ERROR_TYPES.get(str(data.get("error")), HarnessError)
    return cls(str(data["code"]), str(data.get("message", "")), evidence=dict(data.get("evidence") or {}))
