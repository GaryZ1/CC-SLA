"""Ingestion records and configuration models (plan §4.1 to §4.5).

Configuration is validated with Pydantic (unknown keys rejected). Evidence records are plain frozen
dataclasses: ingestion keeps provenance per cell and never coerces types through a DataFrame reader.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from harness.errors import ConfigError
from harness.spec.common import Frozen, Severity

EXTRACTION_VERSION = "ingest/0.1.0"


# --------------------------------------------------------------------------- limits (config/ingest_limits.yaml)


class IngestLimits(Frozen):
    max_file_bytes: int = Field(gt=0)
    max_uncompressed_bytes: int = Field(gt=0)
    max_zip_entries: int = Field(gt=0)
    max_compression_ratio: int = Field(gt=0)
    max_sheets: int = Field(gt=0)
    max_rows_per_sheet: int = Field(gt=0)
    max_columns_per_sheet: int = Field(gt=0)
    max_cell_chars: int = Field(gt=0)
    parse_timeout_seconds: float = Field(gt=0)
    parse_memory_mb: int = Field(gt=0)


class InclusionRules(Frozen):
    hidden_sheets: Literal["extract_and_flag", "exclude_and_report"]
    very_hidden_sheets: Literal["extract_and_flag", "exclude_and_report"]
    hidden_rows: Literal["extract_and_flag", "exclude_and_report"]
    hidden_columns: Literal["extract_and_flag", "exclude_and_report"]
    comments: Literal["extract"]
    hyperlinks: Literal["record_target_only"]
    external_links: Literal["record_only"]


class IngestConfig(Frozen):
    schema_version: Literal["ingest_limits/1.0.0"]
    limits: IngestLimits
    inclusion: InclusionRules


# --------------------------------------------------------------------------- mapping (config/column_mapping.yaml)


class FieldType(StrEnum):
    IDENTIFIER = "identifier"
    TEXT = "text"
    CODE = "code"


class FieldMapping(Frozen):
    aliases: list[str] = Field(min_length=1)
    type: FieldType
    required: bool
    missing_severity: Severity | None = None   # issue when no column maps to this field (optional fields)
    empty_severity: Severity | None = None     # issue when a mapped cell is EMPTY (optional fields)


class FieldOverride(Frozen):
    aliases: list[str] = Field(min_length=1)


class SheetOverride(Frozen):
    header_row: int | None = Field(default=None, ge=1)
    header_rows: list[int] | None = None
    fields: dict[str, FieldOverride] = {}


class SheetSelection(Frozen):
    include: list[str] = ["*"]
    exclude: list[str] = []
    overrides: dict[str, SheetOverride] = {}


class MappingDefaults(Frozen):
    header_search_rows: str = Field(pattern=r"^[1-9][0-9]*-[1-9][0-9]*$")
    multirow_header_joiner: str
    empty_values: list[str]
    code_dialect_candidates: list[str] = Field(min_length=1)

    @property
    def header_search_range(self) -> range:
        low, high = (int(part) for part in self.header_search_rows.split("-"))
        return range(low, high + 1)


class ColumnMapping(Frozen):
    schema_version: Literal["column_mapping/1.0.0"]
    defaults: MappingDefaults
    fields: dict[str, FieldMapping]
    sheets: SheetSelection = SheetSelection()

    @field_validator("fields")
    @classmethod
    def _names_are_identifiers(cls, value: dict[str, FieldMapping]) -> dict[str, FieldMapping]:
        for name in value:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"logical field name {name!r} must be snake_case")
        return value

    def sheet_included(self, sheet: str) -> bool:
        included = any(fnmatch.fnmatchcase(sheet, pattern) for pattern in self.sheets.include)
        excluded = any(fnmatch.fnmatchcase(sheet, pattern) for pattern in self.sheets.exclude)
        return included and not excluded

    def fields_for_sheet(self, sheet: str) -> dict[str, FieldMapping]:
        """Default fields, with sheet override aliases appended (overrides extend, never replace)."""
        override = self.sheets.overrides.get(sheet)
        if override is None:
            return dict(self.fields)
        merged: dict[str, FieldMapping] = {}
        for name, spec in self.fields.items():
            extra = override.fields.get(name)
            if extra is None:
                merged[name] = spec
            else:
                merged[name] = spec.model_copy(update={"aliases": [*spec.aliases, *extra.aliases]})
        unknown = sorted(set(override.fields) - set(self.fields))
        if unknown:
            raise ConfigError("MAPPING_UNKNOWN_FIELD", f"sheet override {sheet!r} names unknown fields {unknown}")
        return merged


def _load_yaml_model(path: Path, model: type[Frozen]) -> Any:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("CONFIG_NOT_FOUND", str(path)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError("CONFIG_NOT_YAML", f"{path}: {exc}") from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            "CONFIG_INVALID",
            f"{path}: {exc.error_count()} error(s)",
            evidence={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def load_ingest_config(path: Path) -> IngestConfig:
    result: IngestConfig = _load_yaml_model(path, IngestConfig)
    return result


def load_column_mapping(path: Path) -> ColumnMapping:
    result: ColumnMapping = _load_yaml_model(path, ColumnMapping)
    return result


# --------------------------------------------------------------------------- evidence records


@dataclass(frozen=True)
class AdmissionResult:
    sha256: str
    size_bytes: int
    format: str
    has_vba: bool
    vba_sha256: str | None
    has_external_links: bool
    has_connections: bool
    is_xlsb: bool
    zip_entries: int
    uncompressed_bytes: int
    max_compression_ratio: float


@dataclass(frozen=True)
class CellEvidence:
    """One non-empty cell as observed. Formula text and cached value are kept apart."""

    sheet: str
    coordinate: str
    row: int
    column: str
    data_type: str
    raw: Any
    excel_formula: str | None
    cached_value: Any
    cached_value_present: bool | None
    number_format: str
    display_text: str | None
    comment: str | None
    hyperlink_target: str | None
    merged_range: str | None
    is_merged_placeholder: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SheetEvidence:
    name: str
    index: int
    state: str                     # visible | hidden | veryHidden
    max_row: int
    max_column: int
    merged_ranges: list[str]
    hidden_rows: list[int]
    hidden_columns: list[str]
    tables: list[str]
    data_validations: int
    cells: list[CellEvidence]

    def cell_map(self) -> dict[str, CellEvidence]:
        return {cell.coordinate: cell for cell in self.cells}


@dataclass(frozen=True)
class WorkbookEvidence:
    sheets: list[SheetEvidence]
    defined_names: list[str]


@dataclass(frozen=True)
class Issue:
    """An ingestion finding. IDs are assigned once, in a deterministic order, when the run is emitted."""

    severity: Severity
    code: str
    message: str
    sheet: str = ""
    cells: str = ""
    record_id: str = ""
    decision_needed: str = ""
    owner: str = ""
    continue_unrelated: bool = True
    stage: int = 0                 # ordering key: workbook, sheet structure, mapping, rows, notes
    sort_key: tuple[int, int, int] = field(default=(0, 0, 0))


ISSUE_CSV_COLUMNS = (
    "issue_id",
    "severity",
    "code",
    "workbook_sha256",
    "sheet",
    "cells",
    "record_id",
    "message",
    "decision_needed",
    "owner",
    "continue_unrelated",
)
