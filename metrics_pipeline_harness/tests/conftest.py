from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from harness.ingest.extract import IngestResult, ingest_workbook
from harness.ingest.models import ColumnMapping, IngestConfig, load_column_mapping, load_ingest_config

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "fixtures" / "synthetic" / "banking"
SYNTHETIC_WORKBOOK = SYNTHETIC / "SYNTHETIC_retail_kpi_inventory.xlsx"


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests run from the project root so relative config paths resolve the same way as the CLI."""
    monkeypatch.chdir(ROOT)
    for var in [v for v in os.environ if v.startswith("HARNESS_")]:
        monkeypatch.delenv(var)
    yield


@pytest.fixture(scope="session")
def ingest_config() -> IngestConfig:
    return load_ingest_config(ROOT / "config" / "ingest_limits.yaml")


@pytest.fixture(scope="session")
def mapping() -> ColumnMapping:
    return load_column_mapping(ROOT / "config" / "column_mapping.yaml")


@pytest.fixture(scope="session")
def mapping_bytes() -> bytes:
    return (ROOT / "config" / "column_mapping.yaml").read_bytes()


@pytest.fixture(scope="session")
def synthetic_result(ingest_config: IngestConfig, mapping: ColumnMapping, mapping_bytes: bytes) -> IngestResult:
    return ingest_workbook(SYNTHETIC_WORKBOOK, config=ingest_config, mapping=mapping, mapping_bytes=mapping_bytes,
                           display_path="fixtures/synthetic/banking/SYNTHETIC_retail_kpi_inventory.xlsx")


@pytest.fixture
def ingest(ingest_config: IngestConfig, mapping: ColumnMapping, mapping_bytes: bytes):  # type: ignore[no-untyped-def]
    def run(path: Path, *, config: IngestConfig | None = None, column_mapping: ColumnMapping | None = None,
            ) -> IngestResult:
        return ingest_workbook(path, config=config or ingest_config, mapping=column_mapping or mapping,
                               mapping_bytes=mapping_bytes, display_path=path.name)
    return run
