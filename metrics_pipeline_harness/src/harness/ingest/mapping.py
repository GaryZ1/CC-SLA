"""Stages 6-7: header location and column mapping (plan §4.1, §4.4).

Header rows are scored against the mapping aliases. Two-row headers are joined as ``Parent / Child``;
merged header cells are expanded only into header labels, never into data fields. Normalization
(case-fold, punctuation to spaces, collapsed whitespace) is applied to copies used for matching; the
original label text is kept alongside.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from harness.ingest.models import CellEvidence, ColumnMapping, FieldMapping, SheetEvidence


def normalize_label(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    no_punct = re.sub(r"[^\w\s]", " ", folded)
    return " ".join(no_punct.split())


@dataclass(frozen=True)
class ColumnHeader:
    column: str
    label: str                  # original text, parts joined with the configured joiner
    parts: tuple[str, ...]
    normalized: str
    field: str | None


@dataclass(frozen=True)
class HeaderMap:
    sheet: str
    header_rows: tuple[int, ...]
    columns: dict[str, ColumnHeader]
    field_columns: dict[str, str]     # logical field -> column letter
    data_start_row: int

    @property
    def unmapped(self) -> list[ColumnHeader]:
        return [h for h in self.columns.values() if h.field is None]


@dataclass(frozen=True)
class HeaderResult:
    header: HeaderMap | None
    status: str                       # FOUND | NOT_FOUND | AMBIGUOUS | OVERRIDE_INVALID
    detail: str = ""
    candidates: list[tuple[tuple[int, ...], int]] = field(default_factory=list)


def _anchor(rng: str) -> str:
    return rng.split(":")[0]


def _label(cells: dict[str, CellEvidence], coord: str) -> str:
    cell = cells.get(coord)
    if cell is None:
        return ""
    if cell.is_merged_placeholder and cell.merged_range:
        anchor = cells.get(_anchor(cell.merged_range))
        cell = anchor
    if cell is None or cell.raw is None or cell.excel_formula is not None:
        return ""
    return str(cell.raw).strip() if isinstance(cell.raw, str | int | float) else ""


def column_letters(max_column: int) -> list[str]:
    letters: list[str] = []
    for index in range(1, max_column + 1):
        value, out = index, ""
        while value:
            value, rem = divmod(value - 1, 26)
            out = chr(65 + rem) + out
        letters.append(out)
    return letters


def _alias_index(fields: dict[str, FieldMapping]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name, spec in fields.items():
        for alias in spec.aliases:
            index.setdefault(normalize_label(alias), name)
    return index


def _build(sheet: SheetEvidence, rows: tuple[int, ...], fields: dict[str, FieldMapping], joiner: str,
           ) -> tuple[dict[str, ColumnHeader], dict[str, list[str]]]:
    cells = sheet.cell_map()
    aliases = _alias_index(fields)
    columns: dict[str, ColumnHeader] = {}
    matches: dict[str, list[str]] = {}
    for col in column_letters(sheet.max_column):
        parts: list[str] = []
        for row in rows:
            text = _label(cells, f"{col}{row}")
            if text and (not parts or parts[-1] != text):   # a vertical merge repeats its label; keep it once
                parts.append(text)
        if not parts:
            continue
        label = joiner.join(parts)
        normalized = normalize_label(label)
        mapped = aliases.get(normalized)
        columns[col] = ColumnHeader(col, label, tuple(parts), normalized, mapped)
        if mapped:
            matches.setdefault(mapped, []).append(col)
    return columns, matches


def locate_header(sheet: SheetEvidence, mapping: ColumnMapping) -> HeaderResult:
    fields = mapping.fields_for_sheet(sheet.name)
    joiner = mapping.defaults.multirow_header_joiner
    required = {name for name, spec in fields.items() if spec.required}
    override = mapping.sheets.overrides.get(sheet.name)
    if override and (override.header_rows or override.header_row):
        candidates_rows = [tuple(override.header_rows or [override.header_row or 1])]
    else:
        search = [r for r in mapping.defaults.header_search_range if r <= sheet.max_row]
        candidates_rows = [(r,) for r in search] + [(r, r + 1) for r in search if r + 1 <= sheet.max_row]

    scored: list[tuple[int, tuple[int, ...], dict[str, ColumnHeader], dict[str, list[str]]]] = []
    ambiguous: list[tuple[tuple[int, ...], list[str]]] = []
    for rows in candidates_rows:
        columns, matches = _build(sheet, rows, fields, joiner)
        duplicates = sorted(name for name, cols in matches.items() if len(cols) > 1)
        if not required <= set(matches):
            continue
        if duplicates:
            ambiguous.append((rows, duplicates))
            continue
        scored.append((len(matches), rows, columns, matches))

    if not scored:
        if override and (override.header_rows or override.header_row):
            return HeaderResult(None, "OVERRIDE_INVALID", f"configured header rows {candidates_rows[0]} do not map "
                                f"required fields {sorted(required)}")
        if ambiguous:
            rows, dups = ambiguous[0]
            return HeaderResult(None, "AMBIGUOUS", f"rows {list(rows)} map {dups} to more than one column")
        return HeaderResult(None, "NOT_FOUND", "no mapped headers found")

    # Highest score wins; on a tie the candidate with fewer rows, then the earliest, wins.
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    best_score, rows, columns, matches = scored[0]
    signature = {c: h.normalized for c, h in columns.items()}
    rivals = [
        s for s in scored[1:]
        if s[0] == best_score and len(s[1]) == len(rows)
        # the same header printed again further down is a repeated header, not a rival (HDR_REPEATED_IN_DATA)
        and {c: h.normalized for c, h in s[2].items()} != signature
    ]
    if rivals:
        return HeaderResult(None, "AMBIGUOUS", f"header rows {list(rows)} and {list(rivals[0][1])} score equally",
                            [(s[1], s[0]) for s in scored])
    header = HeaderMap(
        sheet=sheet.name,
        header_rows=rows,
        columns=columns,
        field_columns={name: matches[name][0] for name in fields if name in matches},   # mapping order
        data_start_row=max(rows) + 1,
    )
    return HeaderResult(header, "FOUND", candidates=[(s[1], s[0]) for s in scored])


def is_repeated_header(cells: dict[str, CellEvidence], row: int, header: HeaderMap) -> bool:
    """A data row whose mapped cells repeat the header labels (for example a header printed per page)."""
    hits = 0
    for col_header in header.columns.values():
        if col_header.field is None:
            continue
        text = _label(cells, f"{col_header.column}{row}")
        if text and normalize_label(text) in {normalize_label(p) for p in col_header.parts}:
            hits += 1
    mapped = sum(1 for h in header.columns.values() if h.field)
    return mapped > 0 and hits * 2 >= mapped and hits >= 2
