"""Source and join contracts (plan §4.10).

The semantic spec says what a metric means; the source contract says what a source actually
guarantees. Contracts start as PROPOSED (or SYNTHETIC locally) and reach CATALOG_VERIFIED and
PROFILE_VERIFIED only with catalog and profiling evidence (P4-04).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from harness.errors import ValidationFailed
from harness.spec.common import EvidenceStatus, Frozen, Resolvable

SIMPLE_TYPES = {"TEXT", "INTEGER", "BIGINT", "BOOLEAN", "DATE", "TIMESTAMP_NTZ", "TIMESTAMP", "DOUBLE"}
DECIMAL_RE = re.compile(r"^DECIMAL\((\d{1,2}),(\d{1,2})\)$")


def validate_logical_type(value: str) -> str:
    """Logical types are a closed vocabulary; DECIMAL needs precision 1-38 and scale <= precision."""
    if value in SIMPLE_TYPES:
        return value
    match = DECIMAL_RE.match(value.replace(" ", ""))
    if match:
        precision, scale = int(match.group(1)), int(match.group(2))
        if 1 <= precision <= 38 and 0 <= scale <= precision:
            return f"DECIMAL({precision},{scale})"
    raise ValueError(f"unsupported logical type {value!r}")


class ColumnContract(Frozen):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    logical_type: str
    nullable: bool
    domain: list[str] | None = None              # allowed values, if enumerated
    observed_values: list[str] | None = None     # profiled values; evidence, not a constraint
    timezone: str | None = None                  # for timestamps
    meaning: str | None = None
    classification: Resolvable[str]

    @field_validator("logical_type")
    @classmethod
    def _type(cls, value: str) -> str:
        return validate_logical_type(value)


class SourceContract(Frozen):
    schema_version: Literal["source_contract/1.0.0"]
    source_id: str = Field(pattern=r"^[A-Za-z][\w]*(\.[A-Za-z_][\w]*){1,3}$")
    evidence_status: EvidenceStatus
    engine: Resolvable[str]
    owner: Resolvable[str]
    delivery_mode: Literal["SNAPSHOT", "APPEND", "CDC", "MIXED"]
    grain: list[str] = Field(min_length=1)
    business_keys: list[str] = Field(min_length=1)
    ordering: Resolvable[str]
    delete_representation: Resolvable[str]
    event_timestamp: Resolvable[str]
    update_timestamp: Resolvable[str]
    history_available: Resolvable[str]
    extraction_boundary: Resolvable[str]
    fixture: str | None = None                   # SYNTHETIC contracts only: where the fixture records live
    columns: list[ColumnContract] = Field(min_length=1)

    @model_validator(mode="after")
    def _keys_are_columns(self) -> SourceContract:
        names = [c.name for c in self.columns]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate columns {duplicates}")
        missing = sorted((set(self.grain) | set(self.business_keys)) - set(names))
        if missing:
            raise ValueError(f"grain or business keys reference unknown columns {missing}")
        for key in self.business_keys:
            if next(c for c in self.columns if c.name == key).nullable:
                raise ValueError(f"business key column {key!r} must be NOT NULL")
        for attr in ("event_timestamp", "update_timestamp"):
            value = getattr(self, attr)
            if isinstance(value, str) and value not in names:
                raise ValueError(f"{attr} {value!r} is not a column")
        if self.fixture is not None and self.evidence_status is not EvidenceStatus.SYNTHETIC:
            raise ValueError("fixture paths are allowed only on SYNTHETIC contracts")
        return self

    def column(self, name: str) -> ColumnContract:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


class JoinContract(Frozen):
    join_id: str = Field(pattern=r"^JC-[0-9]{3,}$")
    left: str
    right: str
    keys: list[tuple[str, str]] = Field(min_length=1)
    temporal_predicate: str | None
    expected_cardinality: Literal["1:1", "N:1", "1:N", "N:M"]
    unmatched_left: Literal["UNKNOWN_MEMBER", "QUARANTINE", "FAIL"]
    dedup_policy: str | None
    fanout_check_id: str
    evidence_status: EvidenceStatus


def load_source_contract(path: Path) -> SourceContract:
    try:
        return SourceContract.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        raise ValidationFailed(
            "SOURCE_CONTRACT_INVALID",
            f"{path}: {exc.error_count()} error(s)",
            evidence={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def load_source_contracts(directory: Path) -> dict[str, SourceContract]:
    contracts: dict[str, SourceContract] = {}
    for path in sorted(directory.glob("*.yaml")):
        contract = load_source_contract(path)
        if contract.source_id in contracts:
            raise ValidationFailed("SOURCE_CONTRACT_DUPLICATE", f"{contract.source_id} defined twice ({path})")
        contracts[contract.source_id] = contract
    return contracts
