"""Checklist §12.5: the synthetic workbook regenerates byte-identically."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from tests.conftest import SYNTHETIC, SYNTHETIC_WORKBOOK


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("make_workbook", SYNTHETIC / "make_workbook.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regeneration_is_byte_identical() -> None:
    module = _module()
    first, second = module.build(), module.build()
    assert first == second
    recorded = (SYNTHETIC / "SYNTHETIC_retail_kpi_inventory.xlsx.sha256").read_text().split()[0]
    assert hashlib.sha256(first).hexdigest() == recorded
    assert hashlib.sha256(SYNTHETIC_WORKBOOK.read_bytes()).hexdigest() == recorded


def test_verify_flag(tmp_path: Path) -> None:
    module = _module()
    assert module.main(["--verify"]) == 0
    stale = tmp_path / "stale.xlsx"
    stale.write_bytes(b"not the fixture")
    assert module.main(["--verify", "--out", str(stale)]) == 3
