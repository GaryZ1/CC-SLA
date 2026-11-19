"""Stages 3-5: two-pass workbook evidence extraction (plan §4.1, §4.7).

Never evaluates formulas, never runs macros, never refreshes or follows links. ``read_workbook_evidence``
is the entry point used inside the bounded subprocess (``run_bounded``); it returns plain JSON data that
``evidence_from_json`` turns back into typed records in the parent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

from harness.errors import ResourceLimitExceeded, WorkbookCorrupt
from harness.ingest.models import CellEvidence, IngestLimits, SheetEvidence, WorkbookEvidence

EXCEL_ERROR_VALUES = frozenset(
    {"#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A", "#GETTING_DATA", "#SPILL!", "#CALC!"}
)


# --------------------------------------------------------------------------- value encoding


def encode_value(value: Any) -> Any:
    """JSON-safe, type-preserving encoding of a cell value. Strings are never modified."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else {"$float": repr(value)}
    if isinstance(value, dt.datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, dt.date):
        return {"$date": value.isoformat()}
    if isinstance(value, dt.time):
        return {"$time": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {"$timedelta": value.total_seconds()}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    return {"$repr": f"{type(value).__name__}:{value}"}


def _col_letter(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _range_coords(min_row: int, max_row: int, min_col: int, max_col: int) -> Iterator[str]:
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            yield f"{_col_letter(col)}{row}"


def _display_text(value: Any, number_format: str) -> str | None:
    """Reproduce only zero-padding formats such as '000000' for integers; anything else stays None."""
    if isinstance(value, int) and not isinstance(value, bool) and number_format and set(number_format) == {"0"}:
        return str(value).zfill(len(number_format))
    return None


def _formula_text(value: Any) -> str:
    text = getattr(value, "text", None)  # ArrayFormula / DataTableFormula keep their text here
    return str(text if text is not None else value)


# --------------------------------------------------------------------------- subprocess entry point


def read_workbook_evidence(path_str: str, limits_data: dict[str, Any]) -> dict[str, Any]:
    """Two independent loads: formulas as written, and cached values as last saved by Excel."""
    import openpyxl  # imported in the subprocess only; defusedxml is picked up automatically when installed

    limits = IngestLimits.model_validate(limits_data)
    path = Path(path_str)
    common: dict[str, Any] = {"read_only": False, "keep_vba": False, "keep_links": False}
    try:
        wb_formula = openpyxl.load_workbook(path, data_only=False, **common)
        wb_cached = openpyxl.load_workbook(path, data_only=True, **common)
    except (KeyError, ValueError, OSError) as exc:
        raise WorkbookCorrupt("WB_PARSE_FAILED", f"{type(exc).__name__}: {exc}") from exc
    if len(wb_formula.worksheets) > limits.max_sheets:
        raise ResourceLimitExceeded("WB_SHEET_LIMIT", f"{len(wb_formula.worksheets)} sheets > {limits.max_sheets}")

    sheets: list[dict[str, Any]] = []
    for index, ws in enumerate(wb_formula.worksheets):
        if ws.max_row > limits.max_rows_per_sheet or ws.max_column > limits.max_columns_per_sheet:
            raise ResourceLimitExceeded(
                "WB_SHEET_DIMENSIONS", f"{ws.title}: {ws.max_row} rows x {ws.max_column} columns"
            )
        ws_cached = wb_cached[ws.title]
        merged_ranges = sorted(str(rng) for rng in ws.merged_cells.ranges)
        merged_by_cell: dict[str, str] = {}
        for rng in ws.merged_cells.ranges:
            for coord in _range_coords(rng.min_row, rng.max_row, rng.min_col, rng.max_col):
                merged_by_cell[coord] = str(rng)
        hidden_rows = sorted(r for r, dim in ws.row_dimensions.items() if dim.hidden)
        hidden_columns: set[str] = set()
        for dim in ws.column_dimensions.values():
            if dim.hidden:
                low, high = dim.min or 0, dim.max or 0
                hidden_columns.update(_col_letter(i) for i in range(low, high + 1) if i > 0)
        cells: list[dict[str, Any]] = []
        for row in ws.iter_rows():
            for cell in row:
                coord = cell.coordinate
                if type(cell).__name__ == "MergedCell":
                    cells.append(
                        CellEvidence(
                            sheet=ws.title, coordinate=coord, row=cell.row, column=_col_letter(cell.column),
                            data_type="merged", raw=None, excel_formula=None, cached_value=None,
                            cached_value_present=None, number_format="", display_text=None, comment=None,
                            hyperlink_target=None, merged_range=merged_by_cell.get(coord),
                            is_merged_placeholder=True,
                        ).as_dict()
                    )
                    continue
                if cell.value is None and cell.comment is None and cell.hyperlink is None:
                    continue
                is_formula = cell.data_type == "f"
                value = cell.value
                if isinstance(value, str) and len(value) > limits.max_cell_chars:
                    raise ResourceLimitExceeded("WB_CELL_TOO_LONG", f"{ws.title}!{coord}")
                cached = ws_cached[coord].value if is_formula else None
                number_format = str(cell.number_format or "")
                cells.append(
                    CellEvidence(
                        sheet=ws.title,
                        coordinate=coord,
                        row=cell.row,
                        column=_col_letter(cell.column),
                        data_type=str(cell.data_type),
                        raw=None if is_formula else encode_value(value),
                        excel_formula=_formula_text(value) if is_formula else None,
                        cached_value=encode_value(cached),
                        cached_value_present=(cached is not None) if is_formula else None,
                        number_format=number_format,
                        display_text=None if is_formula else _display_text(value, number_format),
                        comment=cell.comment.text if cell.comment else None,
                        hyperlink_target=(cell.hyperlink.target or cell.hyperlink.location) if cell.hyperlink else None,
                        merged_range=merged_by_cell.get(coord),
                    ).as_dict()
                )
        sheets.append(
            {
                "name": ws.title,
                "index": index,
                "state": ws.sheet_state,
                "max_row": ws.max_row,
                "max_column": ws.max_column,
                "merged_ranges": merged_ranges,
                "hidden_rows": hidden_rows,
                "hidden_columns": sorted(hidden_columns, key=lambda c: (len(c), c)),
                "tables": sorted(ws.tables.keys()),
                "data_validations": len(ws.data_validations.dataValidation),
                "cells": cells,
            }
        )
    return {"sheets": sheets, "defined_names": sorted(wb_formula.defined_names.keys())}


def evidence_from_json(data: dict[str, Any]) -> WorkbookEvidence:
    sheets = [
        SheetEvidence(
            name=s["name"],
            index=s["index"],
            state=s["state"],
            max_row=s["max_row"],
            max_column=s["max_column"],
            merged_ranges=list(s["merged_ranges"]),
            hidden_rows=list(s["hidden_rows"]),
            hidden_columns=list(s["hidden_columns"]),
            tables=list(s["tables"]),
            data_validations=int(s["data_validations"]),
            cells=[CellEvidence(**c) for c in s["cells"]],
        )
        for s in data["sheets"]
    ]
    return WorkbookEvidence(sheets=sheets, defined_names=list(data["defined_names"]))
