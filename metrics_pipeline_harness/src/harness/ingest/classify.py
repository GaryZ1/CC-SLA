"""Stages 9-10: cell classification and language detection (plan §4.1, §4.3).

Detection is evidence, not a decision: the dialect of authoritative code comes from the source engine
(A-11, Q-14), never from how a snippet happens to parse. Nothing here executes code; Python is only
parsed with ``ast.parse``.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from harness.ingest.models import CellEvidence, FieldType

logging.getLogger("sqlglot").setLevel(logging.ERROR)


class CellClass(StrEnum):
    ENGLISH_TEXT = "ENGLISH_TEXT"
    EXCEL_FORMULA = "EXCEL_FORMULA"
    CACHED_VALUE = "CACHED_VALUE"
    CODE_STRING = "CODE_STRING"
    EMPTY = "EMPTY"
    ERROR_VALUE = "ERROR_VALUE"
    TYPED_VALUE = "TYPED_VALUE"   # number, date or boolean literal typed into the cell (not a formula)


class Language(StrEnum):
    ENGLISH = "ENGLISH"
    EXCEL_FORMULA = "EXCEL_FORMULA"
    SQL = "SQL"
    PYTHON = "PYTHON"
    DAX = "DAX"
    SAS = "SAS"
    R = "R"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


SQL_STATEMENT_TYPES: tuple[type, ...] = (
    exp.Query,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
)

# Lexical markers that suggest a dialect family. A hint only; it never sets the source dialect.
DIALECT_MARKERS: dict[str, tuple[str, ...]] = {
    "tsql": (r"\bGETDATE\s*\(", r"\bSYSDATETIME\s*\(", r"\bDATEADD\s*\(", r"\bDATEDIFF\s*\(\s*(day|month|year)\b",
             r"\bISNULL\s*\(", r"\bSELECT\s+TOP\s+\d", r"\bNVARCHAR\b", r"\[[A-Za-z_][\w ]*\]\.\["),
    "postgres": (r"::\s*[a-z]", r"\bILIKE\b", r"\bDATE_TRUNC\s*\(\s*'"),
    "snowflake": (r"\bIFF\s*\(", r"\bDATEADD\s*\(\s*'", r"\bQUALIFY\b"),
    "oracle": (r"\bSYSDATE\b", r"\bNVL\s*\(", r"\bFROM\s+DUAL\b"),
}

DAX_PATTERNS = (r"\bCALCULATE\s*\(", r"\bSUMX\s*\(", r"\bDIVIDE\s*\(", r"'[^']+'\[[^\]]+\]", r"^\s*\w[\w ]*:=")
SAS_PATTERNS = (r"(?im)^\s*proc\s+\w+", r"(?im)^\s*data\s+[\w.]+\s*;", r"(?im)^\s*run\s*;")
PYTHON_SIGNAL_NODES = (ast.Call, ast.Attribute, ast.Subscript, ast.Assign, ast.FunctionDef, ast.Import,
                       ast.ImportFrom, ast.Lambda, ast.ListComp)


@dataclass(frozen=True)
class LanguageDetection:
    detected: Language
    dialect: str | None
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected.value,
            "dialect": self.dialect,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def sql_parse_dialects(text: str, dialects: list[str]) -> list[str]:
    """Dialects in which ``text`` parses as one or more SQL statements (bare expressions do not count)."""
    parsed: list[str] = []
    for dialect in dialects:
        try:
            statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
        except (SqlglotError, ValueError, RecursionError):
            continue
        if statements and all(isinstance(s, SQL_STATEMENT_TYPES) for s in statements):
            parsed.append(dialect)
    return parsed


def looks_like_python(text: str) -> bool:
    try:
        tree = ast.parse(text, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return any(isinstance(node, PYTHON_SIGNAL_NODES) for node in ast.walk(tree))


def _dialect_hints(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for dialect, patterns in DIALECT_MARKERS.items():
        found = [m.group(0).strip() for p in patterns if (m := re.search(p, text, flags=re.IGNORECASE))]
        if found:
            hits[dialect] = found
    return hits


def _declared(declared: str | None) -> Language | None:
    if not declared:
        return None
    token = declared.strip().upper()
    aliases = {"T-SQL": "SQL", "TSQL": "SQL", "PY": "PYTHON", "PANDAS": "PYTHON", "EXCEL": "EXCEL_FORMULA"}
    token = aliases.get(token, token)
    return Language(token) if token in Language.__members__ else Language.UNKNOWN


def detect_language(text: str, dialects: list[str], declared: str | None = None) -> LanguageDetection:
    stripped = text.strip()
    declared_lang = _declared(declared)
    if stripped.startswith("="):
        return LanguageDetection(Language.EXCEL_FORMULA, None, 0.6, ["leading '=' in a text cell"])
    sql = sql_parse_dialects(stripped, dialects)
    python = looks_like_python(stripped)
    dax = [p for p in DAX_PATTERNS if re.search(p, stripped)]
    sas = [p for p in SAS_PATTERNS if re.search(p, stripped)]
    evidence: list[str] = []
    if sql and python:
        return LanguageDetection(Language.MIXED, None, 0.2, [f"parses as SQL: {', '.join(sql)}", "parses as Python"])
    if sql:
        evidence.append(f"parses: {', '.join(sql)}")
        hints = _dialect_hints(stripped)
        dialect: str | None = None
        confidence = 0.55
        if len(hints) == 1:
            dialect = next(iter(hints))
            confidence = 0.75
            evidence.append(f"{dialect}-family markers: {', '.join(hints[dialect])}")
        elif hints:
            evidence.append(f"conflicting dialect markers: {sorted(hints)}")
        else:
            evidence.append("no dialect-specific functions or quoting")
        if declared_lang is Language.SQL:
            confidence = round(min(0.9, confidence + 0.15), 2)
            evidence.append("declared language agrees")
        return LanguageDetection(Language.SQL, dialect, confidence, evidence)
    if python:
        confidence = 0.9 if declared_lang is Language.PYTHON else 0.7
        return LanguageDetection(Language.PYTHON, None, confidence, ["ast.parse succeeds; calls or attributes present"])
    if dax:
        return LanguageDetection(Language.DAX, None, 0.5, ["lexical DAX markers (not parsed)"])
    if sas:
        return LanguageDetection(Language.SAS, None, 0.5, ["lexical SAS markers (not parsed)"])
    return LanguageDetection(Language.UNKNOWN, None, 0.0, ["no supported parser accepted the text"])


def is_empty_text(value: Any, empty_values: list[str]) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in empty_values)


def classify_cell(
    cell: CellEvidence | None, field_type: FieldType | None, empty_values: list[str], dialects: list[str]
) -> CellClass:
    if cell is None or cell.is_merged_placeholder:
        return CellClass.EMPTY
    if cell.excel_formula is not None:
        return CellClass.EXCEL_FORMULA
    if cell.data_type == "e":
        return CellClass.ERROR_VALUE
    if is_empty_text(cell.raw, empty_values):
        return CellClass.EMPTY
    if not isinstance(cell.raw, str):
        return CellClass.TYPED_VALUE
    if field_type is FieldType.CODE:
        return CellClass.CODE_STRING
    # A text cell outside a code column is a code string only if it parses as SQL statements or Python.
    if sql_parse_dialects(cell.raw, dialects) or looks_like_python(cell.raw):
        return CellClass.CODE_STRING
    return CellClass.ENGLISH_TEXT
