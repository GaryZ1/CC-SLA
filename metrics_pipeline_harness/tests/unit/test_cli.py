"""CLI contract (plan §12.2): run IDs, artifacts, exit codes, idempotency."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from harness.cli import main
from tests.conftest import SYNTHETIC_WORKBOOK

WB = str(SYNTHETIC_WORKBOOK.relative_to(Path(__file__).resolve().parents[2]))


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HARNESS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    return tmp_path / "artifacts"


def _run_dir(out: str) -> Path:
    line = next(line for line in out.splitlines() if line.startswith("artifacts="))
    return Path(line.split("=", 1)[1])


def test_version() -> None:
    proc = subprocess.run([sys.executable, "-m", "harness.cli", "--version"], capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0 and proc.stdout.strip() == "harness 0.1.0"


def test_extract_writes_four_artifacts_and_manifest(artifacts: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", WB, "--mapping", "config/column_mapping.yaml"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("run_id=")
    run = _run_dir(out)
    names = sorted(p.name for p in run.iterdir())
    assert names == ["column_mapping.yaml", "ingestion_issues.csv", "raw_metric_rows.jsonl", "run.log",
                     "run_manifest.json", "workbook_profile.json"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert set(manifest["artifacts"]) == {"column_mapping.yaml", "ingestion_issues.csv", "raw_metric_rows.jsonl",
                                          "workbook_profile.json"}
    for name, ref in manifest["artifacts"].items():                       # content-addressed copies exist
        digest = ref.removeprefix("sha256:")
        assert (artifacts / "sha256" / digest[:2] / digest).read_bytes() == (run / name).read_bytes()
    log_lines = [json.loads(line) for line in (run / "run.log").read_text().splitlines()]
    assert {line["event"] for line in log_lines} >= {"run_started", "ingest_completed", "run_finished"}
    assert all(line["run_id"] == manifest["run_id"] for line in log_lines)


def test_extract_is_byte_identical_across_runs(artifacts: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", WB]) == 0
    first = json.loads((_run_dir(capsys.readouterr().out) / "run_manifest.json").read_text())
    assert main(["extract", WB]) == 0
    second = json.loads((_run_dir(capsys.readouterr().out) / "run_manifest.json").read_text())
    assert first["run_id"] != second["run_id"]
    assert first["artifacts"] == second["artifacts"]


def test_profile_writes_profile_only(artifacts: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", WB]) == 0
    run = _run_dir(capsys.readouterr().out)
    assert sorted(p.name for p in run.iterdir()) == ["run.log", "run_manifest.json", "workbook_profile.json"]


def test_exit_codes_for_bad_inputs(tmp_path: Path, artifacts: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    assert main(["extract", str(corrupt)]) == 3
    bomb = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/worksheets/sheet1.xml", b"\x00" * (20 * 1024 * 1024))
    assert main(["extract", str(bomb)]) == 7
    assert main(["extract", str(tmp_path / "missing.xlsx")]) == 3
    assert main(["--config", str(tmp_path / "nope.yaml"), "extract", WB]) == 2
    err = capsys.readouterr().err
    assert "WB_CORRUPT" in err and "WB_COMPRESSION_RATIO" in err and "CONFIG_NOT_FOUND" in err


def test_forbidden_env_override_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_ALLOWED_WRITE_TARGETS", "prod.gold")
    assert main(["config", "check"]) == 2


def test_config_check_lists_unknowns(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "check"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{"):])
    assert summary["platform"] == "databricks" and summary["authority_policy"] == "CODE"
    assert summary["unknown_count"] == len(summary["unknown"]) > 20


def test_export_schemas_check_detects_drift(tmp_path: Path) -> None:
    assert main(["export-schemas", "--check"]) == 0
    assert main(["export-schemas", "--dir", str(tmp_path), "--check"]) == 3
    assert main(["export-schemas", "--dir", str(tmp_path)]) == 0
    assert main(["export-schemas", "--dir", str(tmp_path), "--check"]) == 0
