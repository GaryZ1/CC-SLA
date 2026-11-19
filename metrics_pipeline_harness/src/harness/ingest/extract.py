"""Workbook ingestion pipeline: stages 1-12 of plan §4.1.

``ingest_workbook`` is pure with respect to its inputs: the same workbook bytes, configuration and
software versions produce byte-identical artifacts (``workbook_profile.json``, ``raw_metric_rows.jsonl``,
``ingestion_issues.csv`` and the ``column_mapping.yaml`` that was applied). Run IDs and timings live only
in the run manifest and log, never in those artifacts.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from harness.ingest.admit import admit
from harness.ingest.bounded import run_bounded
from harness.ingest.classify import CellClass, Language, classify_cell, detect_language
from harness.ingest.mapping import HeaderMap, is_repeated_header, locate_header
from harness.ingest.models import (
    EXTRACTION_VERSION,
    ISSUE_CSV_COLUMNS,
    AdmissionResult,
    CellEvidence,
    ColumnMapping,
    FieldType,
    IngestConfig,
    Issue,
    SheetEvidence,
    WorkbookEvidence,
)
from harness.ingest.reader import EXCEL_ERROR_VALUES, evidence_from_json, read_workbook_evidence
from harness.ingest.untrusted import excerpt, scan_text
from harness.spec.common import Severity

# Issue ordering stages: workbook and sheet structure first, then content scans, mapping, rows, notes.
STAGE_DISCOVERY, STAGE_UNTRUSTED, STAGE_MAPPING, STAGE_ROWS, STAGE_NOTES = 1, 2, 3, 4, 5
SEVERITY_ORDER = [Severity.BLOCKER, Severity.MAJOR, Severity.MINOR, Severity.INFO]


@dataclass
class IngestResult:
    admission: AdmissionResult
    profile: dict[str, Any]
    records: list[dict[str, Any]]
    issues: list[tuple[str, Issue]]
    mapping_bytes: bytes
    sheets_extracted: list[str] = field(default_factory=list)

    @property
    def blocker_count(self) -> int:
        return sum(1 for _, issue in self.issues if issue.severity is Severity.BLOCKER)

    def profile_bytes(self) -> bytes:
        return (json.dumps(self.profile, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    def records_bytes(self) -> bytes:
        lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in self.records]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def issues_bytes(self) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(ISSUE_CSV_COLUMNS)
        for issue_id, issue in self.issues:
            writer.writerow([
                issue_id, issue.severity.value, issue.code, self.admission.sha256, issue.sheet, issue.cells,
                issue.record_id, issue.message, issue.decision_needed, issue.owner,
                "true" if issue.continue_unrelated else "false",
            ])
        return buffer.getvalue().encode("utf-8")


def _col_index(letters: str) -> int:
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - 64)
    return value


def _record_id(sha: str, sheet: str, row: int) -> str:
    return f"{sha}:{sheet}:{row}"


class _Collector:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, stage: int, sheet_index: int, row: int, col: int, **kwargs: Any) -> None:
        self.issues.append(Issue(stage=stage, sort_key=(sheet_index, row, col), **kwargs))

    def numbered(self) -> list[tuple[str, Issue]]:
        ordered = sorted(
            enumerate(self.issues), key=lambda pair: (pair[1].stage, pair[1].sort_key, pair[0])
        )
        return [(f"ISS-{n:04d}", issue) for n, (_, issue) in enumerate(ordered, start=1)]


def _field_payload(
    cell: CellEvidence | None, coord: str, cls: CellClass, header_label: str, header_parts: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"cell": coord, "class": cls.value}
    if header_parts > 1:
        payload["header_path"] = header_label
    if cell is None or cell.is_merged_placeholder:
        payload["raw"] = None
        return payload
    if cell.excel_formula is not None:
        payload["excel_formula"] = cell.excel_formula
        payload["cached_value"] = cell.cached_value
        payload["cached_value_present"] = cell.cached_value_present
    else:
        payload["raw"] = cell.raw
    if cell.display_text is not None:
        payload["display_text"] = cell.display_text
    if cell.comment is not None:
        payload["comment"] = cell.comment
    if cell.hyperlink_target is not None:
        payload["hyperlink_target"] = cell.hyperlink_target
    return payload


def _discovery_issues(admission: AdmissionResult, evidence: WorkbookEvidence, issues: _Collector) -> None:
    if admission.has_vba:
        issues.add(STAGE_DISCOVERY, -1, 0, 0, severity=Severity.INFO, code="WB_VBA_PRESENT",
                   message=f"VBA project present (sha256 {admission.vba_sha256}); recorded, never executed",
                   decision_needed="Review any macro a metric depends on as text only")
    if admission.has_external_links:
        issues.add(STAGE_DISCOVERY, -1, 0, 1, severity=Severity.INFO, code="WB_EXTERNAL_LINKS",
                   message="External workbook links present; recorded, never refreshed",
                   decision_needed="Confirm no metric depends on linked workbooks")
    if admission.has_connections:
        issues.add(STAGE_DISCOVERY, -1, 0, 2, severity=Severity.INFO, code="WB_DATA_CONNECTIONS",
                   message="Data connections present; recorded, never refreshed")


def _scan_untrusted(sheet: SheetEvidence, data_rows: set[int], sha: str, issues: _Collector) -> None:
    for cell in sheet.cells:
        sources: list[tuple[str, str]] = []
        if isinstance(cell.raw, str):
            sources.append(("Cell text", cell.raw))
        if cell.comment:
            sources.append(("Comment text", cell.comment))
        if cell.hyperlink_target:
            sources.append(("Hyperlink target", cell.hyperlink_target))
        for label, text in sources:
            hits = scan_text(text)
            if not hits:
                continue
            issues.add(
                STAGE_UNTRUSTED, sheet.index, cell.row, _col_index(cell.column),
                severity=Severity.MAJOR, code="UNTRUSTED_INSTRUCTION_TEXT", sheet=sheet.name, cells=cell.coordinate,
                record_id=_record_id(sha, sheet.name, cell.row) if cell.row in data_rows else "",
                message=f"{label} resembles an instruction: '{excerpt(text)}' Treated as data; no effect on policy "
                        f"(patterns: {', '.join(hits)})",
                decision_needed="Confirm the text can be removed from the source workbook",
            )


def _sheet_rows(
    sheet: SheetEvidence,
    header: HeaderMap,
    mapping: ColumnMapping,
    config: IngestConfig,
    sha: str,
    issues: _Collector,
) -> list[dict[str, Any]]:
    cells = sheet.cell_map()
    fields = mapping.fields_for_sheet(sheet.name)
    empty_values = mapping.defaults.empty_values
    dialects = mapping.defaults.code_dialect_candidates
    hidden_rows = set(sheet.hidden_rows)
    sheet_hidden = sheet.state != "visible"
    mapped_cols = {h.column for h in header.columns.values() if h.field}
    records: list[dict[str, Any]] = []
    cells_by_row: dict[int, dict[str, CellEvidence]] = {}
    for cell_evidence in sheet.cells:
        cells_by_row.setdefault(cell_evidence.row, {})[cell_evidence.column] = cell_evidence

    for name, spec in fields.items():
        if name not in header.field_columns and spec.missing_severity is not None:
            first = header.header_rows[0]
            issues.add(
                STAGE_MAPPING, sheet.index, first, 0, severity=spec.missing_severity,
                code="MAP_REQUIRED_FIELD_MISSING", sheet=sheet.name,
                cells=f"A{first}:{_letters(sheet.max_column)}{header.header_rows[-1]}",
                message=f"No {name} column on sheet '{sheet.name}'; records use registry fingerprints for identity",
                decision_needed=f"Supply a {name} column or confirm identity by fingerprint",
            )
    for col_header in header.unmapped:
        issues.add(
            STAGE_NOTES, sheet.index, header.header_rows[0], _col_index(col_header.column), severity=Severity.INFO,
            code="MAP_COLUMN_UNMAPPED", sheet=sheet.name,
            cells=f"{col_header.column}{header.header_rows[0]}",
            message=f"Column '{col_header.label}' is not mapped; values kept under extra with coordinates",
        )

    for row in range(header.data_start_row, sheet.max_row + 1):
        row_cells = cells_by_row.get(row, {})
        has_content = any(
            not c.is_merged_placeholder
            and (c.raw is not None or c.excel_formula is not None or c.comment is not None or c.hyperlink_target)
            for c in row_cells.values()
        )
        if not has_content:
            continue
        if is_repeated_header(cells, row, header):
            issues.add(STAGE_ROWS, sheet.index, row, 0, severity=Severity.MINOR, code="HDR_REPEATED_IN_DATA",
                       sheet=sheet.name, cells=f"A{row}:{_letters(sheet.max_column)}{row}",
                       message=f"Row {row} repeats the header; not extracted")
            continue
        record_id = _record_id(sha, sheet.name, row)
        row_hidden = row in hidden_rows
        if row_hidden and config.inclusion.hidden_rows == "exclude_and_report":
            issues.add(STAGE_DISCOVERY, sheet.index, row, 0, severity=Severity.INFO, code="WB_HIDDEN_CONTENT",
                       sheet=sheet.name, cells=f"A{row}:{_letters(sheet.max_column)}{row}",
                       message=f"Hidden row {row} ({_row_label(cells, header, row)}) excluded by inclusion policy")
            continue
        field_payloads: dict[str, Any] = {}
        merged_context: dict[str, Any] = {}
        excel_formulas: list[dict[str, Any]] = []
        owner_text = ""
        owner_col = header.field_columns.get("owner")
        if owner_col and isinstance(getattr(cells.get(f"{owner_col}{row}"), "raw", None), str):
            owner_text = str(cells[f"{owner_col}{row}"].raw).strip()
        declared = None
        lang_col = header.field_columns.get("code_language")
        if lang_col and isinstance(getattr(cells.get(f"{lang_col}{row}"), "raw", None), str):
            declared = str(cells[f"{lang_col}{row}"].raw)

        for name, column in header.field_columns.items():
            spec = fields[name]
            coord = f"{column}{row}"
            cell = cells.get(coord)
            cls = classify_cell(cell, spec.type, empty_values, dialects)
            col_header = header.columns[column]
            payload = _field_payload(cell, coord, cls, col_header.label, len(col_header.parts))
            if cell is not None and cell.is_merged_placeholder and cell.merged_range:
                anchor = cells.get(cell.merged_range.split(":")[0])
                merged_context[name] = {"range": cell.merged_range, "anchor_value": anchor.raw if anchor else None}
                issues.add(STAGE_ROWS, sheet.index, row, _col_index(column), severity=Severity.MINOR,
                           code="ROW_MERGED_BUSINESS_FIELD", sheet=sheet.name, cells=coord, record_id=record_id,
                           owner=owner_text,
                           message=f"{coord} is covered by merged range {cell.merged_range}; anchor value not copied "
                                   f"into {name}",
                           decision_needed=f"Confirm whether {name} of row {row} equals the merged anchor value")
            if spec.type is FieldType.IDENTIFIER and cell is not None and cell.display_text is not None:
                issues.add(STAGE_ROWS, sheet.index, row, _col_index(column), severity=Severity.INFO,
                           code="ID_NUMERIC_WITH_PAD_FORMAT", sheet=sheet.name, cells=coord, record_id=record_id,
                           owner=owner_text,
                           message=f"{coord} is the number {cell.raw!r} shown as {cell.display_text!r} by format "
                                   f"{cell.number_format!r}; both kept",
                           decision_needed=f"Confirm whether {name} is {cell.display_text!r} (display) or "
                                           f"{cell.raw!r} (stored)")
            if cls is CellClass.CODE_STRING and isinstance(payload.get("raw"), str):
                detection = detect_language(payload["raw"], dialects, declared)
                payload["language"] = detection.as_dict()
                if detection.detected in (Language.UNKNOWN, Language.MIXED):
                    issues.add(STAGE_ROWS, sheet.index, row, _col_index(column), severity=Severity.MINOR,
                               code="LANG_UNKNOWN" if detection.detected is Language.UNKNOWN else "LANG_MIXED",
                               sheet=sheet.name, cells=coord, record_id=record_id, owner=owner_text,
                               message=f"Language of {coord} could not be determined: {'; '.join(detection.evidence)}",
                               decision_needed="Declare the language of this code")
            if cls is CellClass.EXCEL_FORMULA and cell is not None:
                excel_formulas.append({"cell": coord, "formula": cell.excel_formula,
                                       "cached_value": cell.cached_value,
                                       "cached_value_present": cell.cached_value_present})
                _formula_issues(cell, sheet, record_id, owner_text, issues)
            if cls is CellClass.ERROR_VALUE:
                issues.add(STAGE_ROWS, sheet.index, row, _col_index(column), severity=Severity.MINOR,
                           code="WB_FORMULA_ERROR_VALUE", sheet=sheet.name, cells=coord, record_id=record_id,
                           owner=owner_text, message=f"{coord} holds an error value {cell.raw if cell else ''!r}")
            if cls is CellClass.EMPTY and name not in merged_context:
                severity = Severity.MAJOR if spec.required else spec.empty_severity
                if severity is not None:
                    label = _row_label(cells, header, row)
                    if spec.type is FieldType.CODE:
                        formula_col = header.field_columns.get("formula_text")
                        has_formula = formula_col is not None and classify_cell(
                            cells.get(f"{formula_col}{row}"), FieldType.TEXT, empty_values, dialects
                        ) is not CellClass.EMPTY
                        message = f"No code for {label}; " + ("English formula only" if has_formula else
                                                              "no formula either")
                        decision = "Supply code, or confirm the English formula is the definition to implement"
                    else:
                        message = f"Required field {name} is empty for {label}"
                        decision = f"Supply {name} for {label}"
                    issues.add(STAGE_ROWS, sheet.index, row, _col_index(column), severity=severity,
                               code="ROW_EMPTY_REQUIRED", sheet=sheet.name, cells=coord, record_id=record_id,
                               owner=owner_text, message=message, decision_needed=decision)
            field_payloads[name] = payload

        extra: dict[str, Any] = {}
        for column, cell in sorted(row_cells.items(), key=lambda item: _col_index(item[0])):
            if column in mapped_cols or cell.is_merged_placeholder:
                continue
            if cell.raw is None and cell.excel_formula is None and cell.comment is None:
                continue
            extra_header = header.columns.get(column)
            key = extra_header.label if extra_header else f"column_{column}"
            extra[key] = _field_payload(cell, cell.coordinate, classify_cell(cell, None, empty_values, dialects),
                                        key, 1)
            if cell.excel_formula is not None:
                _formula_issues(cell, sheet, record_id, owner_text, issues)

        if row_hidden or sheet_hidden:
            what = "Hidden row" if row_hidden else f"Row on {sheet.state} sheet"
            issues.add(STAGE_DISCOVERY, sheet.index, row, 0, severity=Severity.INFO, code="WB_HIDDEN_CONTENT",
                       sheet=sheet.name, cells=f"A{row}:{_letters(sheet.max_column)}{row}", record_id=record_id,
                       owner=owner_text,
                       message=f"{what} {row} ({_row_label(cells, header, row)}) extracted and flagged",
                       decision_needed=f"Include row {row} in scope?")

        records.append({
            "record_id": record_id,
            "workbook_sha256": sha,
            "extraction_version": EXTRACTION_VERSION,
            "sheet": sheet.name,
            "sheet_state": sheet.state,
            "row": row,
            "row_hidden": row_hidden,
            "requires_owner_confirmation": row_hidden or sheet_hidden,
            "fields": field_payloads,
            "extra": extra,
            "merged_context": merged_context or None,
            "excel_formulas": excel_formulas,
            "issues": [],
        })
    return records


def _duplicate_business_ids(records: list[dict[str, Any]], sheet_index: dict[str, int], issues: _Collector) -> None:
    """WB-07: duplicated business IDs are reported on every row involved and never merged."""
    by_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        field_payload = record["fields"].get("business_id")
        if not field_payload or field_payload["class"] == CellClass.EMPTY.value:
            continue
        value = field_payload.get("display_text") or field_payload.get("raw")
        by_id.setdefault(str(value).strip(), []).append(record)
    for business_id, group in sorted(by_id.items()):
        if len(group) < 2:
            continue
        locations = ", ".join(f"{r['sheet']}!{r['fields']['business_id']['cell']}" for r in group)
        for record in group:
            issues.add(STAGE_ROWS, sheet_index[record["sheet"]], record["row"], 1, severity=Severity.MAJOR,
                       code="ID_DUPLICATE_BUSINESS_ID", sheet=record["sheet"],
                       cells=record["fields"]["business_id"]["cell"], record_id=record["record_id"],
                       message=f"Business ID {business_id} appears {len(group)} times ({locations}); never merged",
                       decision_needed=f"Which row is {business_id}? Rows stay NEEDS_CLARIFICATION until decided")


def _formula_issues(cell: CellEvidence, sheet: SheetEvidence, record_id: str, owner: str, issues: _Collector) -> None:
    col = _col_index(cell.column)
    if not cell.cached_value_present:
        issues.add(STAGE_ROWS, sheet.index, cell.row, col, severity=Severity.INFO, code="WB_CACHED_VALUE_ABSENT",
                   sheet=sheet.name, cells=cell.coordinate, record_id=record_id, owner=owner,
                   message=f"Formula {cell.excel_formula!r} has no cached value; formulas are never recalculated")
    elif isinstance(cell.cached_value, str) and cell.cached_value in EXCEL_ERROR_VALUES:
        issues.add(STAGE_ROWS, sheet.index, cell.row, col, severity=Severity.MINOR, code="WB_FORMULA_ERROR_VALUE",
                   sheet=sheet.name, cells=cell.coordinate, record_id=record_id, owner=owner,
                   message=f"Formula {cell.excel_formula!r} cached an error value {cell.cached_value!r}")


def _letters(index: int) -> str:
    out = ""
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out or "A"


def _row_label(cells: dict[str, CellEvidence], header: HeaderMap, row: int) -> str:
    parts: list[str] = []
    for name in ("business_id", "name"):
        column = header.field_columns.get(name)
        cell = cells.get(f"{column}{row}") if column else None
        if cell is not None and cell.raw is not None and not cell.is_merged_placeholder:
            parts.append(str(cell.raw) if name == "business_id" else f"'{cell.raw}'")
    return " ".join(parts) or f"row {row}"


def ingest_workbook(
    path: Path,
    *,
    config: IngestConfig,
    mapping: ColumnMapping,
    mapping_bytes: bytes,
    display_path: str | None = None,
) -> IngestResult:
    admission = admit(path, config.limits)
    raw = run_bounded(
        read_workbook_evidence,
        str(path),
        config.limits.model_dump(mode="json"),
        timeout_s=config.limits.parse_timeout_seconds,
        memory_mb=config.limits.parse_memory_mb,
    )
    evidence = evidence_from_json(raw)
    sha = admission.sha256
    issues = _Collector()
    _discovery_issues(admission, evidence, issues)

    records: list[dict[str, Any]] = []
    sheet_profiles: list[dict[str, Any]] = []
    extracted: list[str] = []
    for sheet in evidence.sheets:
        profile: dict[str, Any] = {
            "name": sheet.name,
            "state": sheet.state,
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
        }
        comment_cells = [c.coordinate for c in sheet.cells if c.comment]
        formula_cells = sum(1 for c in sheet.cells if c.excel_formula is not None)
        structure = {
            "tables": sheet.tables,
            "merged_ranges": sheet.merged_ranges,
            "hidden_rows": sheet.hidden_rows,
            "hidden_columns": sheet.hidden_columns,
            "comment_cells": comment_cells,
            "hyperlink_cells": [c.coordinate for c in sheet.cells if c.hyperlink_target],
            "formula_cells": formula_cells,
            "data_validations": sheet.data_validations,
        }
        data_rows: set[int] = set()
        if not mapping.sheet_included(sheet.name):
            profile.update(header_rows=[], classification="EXCLUDED_BY_CONFIG", **structure)
            issues.add(STAGE_DISCOVERY, sheet.index, 0, 0, severity=Severity.INFO, code="SHEET_EXCLUDED",
                       sheet=sheet.name, message=f"Sheet '{sheet.name}' excluded by column_mapping.yaml; profiled only")
            _scan_untrusted(sheet, data_rows, sha, issues)
            sheet_profiles.append(profile)
            continue
        hidden_policy = (config.inclusion.very_hidden_sheets if sheet.state == "veryHidden"
                         else config.inclusion.hidden_sheets)
        result = locate_header(sheet, mapping)
        header = result.header
        if header is None:
            profile.update(header_rows=[], classification="NOT_A_METRIC_SHEET", reason=result.detail, **structure)
            if sheet.state != "visible":
                issues.add(STAGE_DISCOVERY, sheet.index, 0, 0, severity=Severity.INFO, code="WB_HIDDEN_CONTENT",
                           sheet=sheet.name,
                           message=f"{sheet.state} sheet '{sheet.name}' has no metric headers; profiled, not extracted",
                           decision_needed="None unless the sheet holds definitions")
            else:
                severity = Severity.MAJOR if result.status in ("AMBIGUOUS", "OVERRIDE_INVALID") else Severity.INFO
                code = "HDR_AMBIGUOUS" if result.status == "AMBIGUOUS" else "HDR_NOT_FOUND"
                issues.add(STAGE_MAPPING, sheet.index, 0, 0, severity=severity, code=code, sheet=sheet.name,
                           message=f"Sheet '{sheet.name}': {result.detail}; profiled, not extracted",
                           decision_needed="Add a sheet override to column_mapping.yaml if this sheet holds metrics")
            _scan_untrusted(sheet, data_rows, sha, issues)
            sheet_profiles.append(profile)
            continue
        if sheet.state != "visible" and hidden_policy == "exclude_and_report":
            profile.update(header_rows=list(header.header_rows), classification="EXCLUDED_HIDDEN", **structure)
            issues.add(STAGE_DISCOVERY, sheet.index, 0, 0, severity=Severity.INFO, code="WB_HIDDEN_CONTENT",
                       sheet=sheet.name, message=f"{sheet.state} sheet '{sheet.name}' excluded by inclusion policy")
            _scan_untrusted(sheet, data_rows, sha, issues)
            sheet_profiles.append(profile)
            continue
        sheet_records = _sheet_rows(sheet, header, mapping, config, sha, issues)
        data_rows = {r["row"] for r in sheet_records}
        _scan_untrusted(sheet, data_rows, sha, issues)
        if sheet.state != "visible":
            issues.add(STAGE_DISCOVERY, sheet.index, 0, 0, severity=Severity.INFO, code="WB_HIDDEN_CONTENT",
                       sheet=sheet.name, message=f"{sheet.state} sheet '{sheet.name}' extracted and flagged",
                       decision_needed=f"Include sheet '{sheet.name}' in scope?")
        for column in sheet.hidden_columns:
            if column in header.columns:
                issues.add(STAGE_DISCOVERY, sheet.index, 0, _col_index(column), severity=Severity.INFO,
                           code="WB_HIDDEN_CONTENT", sheet=sheet.name, cells=f"{column}:{column}",
                           message=f"Hidden column {column} ('{header.columns[column].label}') extracted and flagged",
                           decision_needed=f"Confirm column {column} is in scope")
        first_data = header.data_start_row
        last_data = max(data_rows) if data_rows else first_data - 1
        profile.update(
            header_rows=list(header.header_rows),
            data_rows=f"{first_data}-{last_data}" if data_rows else "",
            field_columns=header.field_columns,
            **structure,
        )
        sheet_profiles.append(profile)
        records.extend(sheet_records)
        extracted.append(sheet.name)

    _duplicate_business_ids(records, {s.name: s.index for s in evidence.sheets}, issues)
    numbered = issues.numbered()
    by_record: dict[str, list[str]] = {}
    for issue_id, issue in numbered:
        if issue.record_id:
            by_record.setdefault(issue.record_id, []).append(issue_id)
    for record in records:
        record["issues"] = by_record.get(record["record_id"], [])

    counts = {sev.value: 0 for sev in SEVERITY_ORDER}
    for _, issue in numbered:
        counts[issue.severity.value] += 1
    profile_doc = {
        "schema_version": "workbook_profile/1.0.0",
        "workbook": {
            "path": display_path if display_path is not None else path.as_posix(),
            "sha256": sha,
            "size_bytes": admission.size_bytes,
            "format": admission.format,
            "has_vba": admission.has_vba,
            "vba_sha256": admission.vba_sha256,
            "has_external_links": admission.has_external_links,
            "has_connections": admission.has_connections,
            "zip_entries": admission.zip_entries,
            "uncompressed_bytes": admission.uncompressed_bytes,
            "max_compression_ratio": admission.max_compression_ratio,
            "extraction_version": EXTRACTION_VERSION,
            "parser": {"name": "openpyxl", "version": version("openpyxl")},
        },
        "sheets": sheet_profiles,
        "defined_names": evidence.defined_names,
        "records_extracted": len(records),
        "issue_counts": counts,
    }
    return IngestResult(admission, profile_doc, records, numbered, mapping_bytes, extracted)
