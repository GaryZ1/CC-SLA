"""Primitives shared by every specification model (plan §4.8).

Three kinds of "missing" are kept apart:

* ``Unknown``       - should have a value; evidence not yet available. May block readiness.
* ``NotApplicable`` - the concept does not apply; never blocks; the justification is reviewed.
* business NULL     - a real, meaningful absence; plain ``None`` on fields declared nullable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    """Immutable model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Unknown(Frozen):
    kind: Literal["UNKNOWN"] = "UNKNOWN"
    reason: str = Field(min_length=1)
    question: str | None = Field(default=None, pattern=r"^Q-[0-9]{2}[a-z]?$")
    issue_id: str | None = None


class NotApplicable(Frozen):
    kind: Literal["NOT_APPLICABLE"] = "NOT_APPLICABLE"
    justification: str = Field(min_length=1)


T = TypeVar("T")

# Union order matters for pydantic's smart mode only when the payload is ambiguous; the two marker
# models are dicts with a literal ``kind`` so they never collide with scalar values.
Resolvable: TypeAlias = T | Unknown | NotApplicable


def is_unknown(value: object) -> bool:
    return isinstance(value, Unknown)


class EvidenceStatus(StrEnum):
    ORIGINAL_ONLY = "ORIGINAL_ONLY"
    PROPOSED = "PROPOSED"
    OWNER_CONFIRMED = "OWNER_CONFIRMED"
    CATALOG_VERIFIED = "CATALOG_VERIFIED"
    PROFILE_VERIFIED = "PROFILE_VERIFIED"
    SYNTHETIC = "SYNTHETIC"


class Severity(StrEnum):
    BLOCKER = "BLOCKER"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"
