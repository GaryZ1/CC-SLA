"""P0-02: project config model and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.config import ENV_OVERRIDES, dump_config, load_config, parse_config, unknown_fields
from harness.errors import ConfigError, ExitCode

CONFIG = Path("config/project.yaml")


def test_loads_and_round_trips_byte_identically() -> None:
    config = load_config(CONFIG, environ={})
    assert dump_config(config) == CONFIG.read_text(encoding="utf-8")
    assert parse_config(yaml.safe_load(dump_config(config))) == config


def test_decisions_d01_to_d03_are_recorded() -> None:
    config = load_config(CONFIG, environ={})
    assert [d.id for d in config.decisions] == ["D-01", "D-02", "D-03"]
    assert config.production.platform == "databricks"
    assert config.production.sql_dialect == "databricks"
    assert config.semantics.authoritative_definition_policy.value == "CODE"
    assert config.sources.location == "cloud_database"


def test_unknown_values_round_trip_unchanged() -> None:
    config = load_config(CONFIG, environ={})
    unknowns = unknown_fields(config)
    paths = {p for p, _, _ in unknowns}
    assert "sources.engine" in paths and "production.unity_catalog" in paths
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["sources"]["engine"] == {
        "kind": "UNKNOWN", "reason": "Database engine and product not stated", "question": "Q-14", "issue_id": None,
    }
    assert {q for _, _, q in unknowns} >= {"Q-01a", "Q-02a", "Q-03", "Q-04", "Q-05", "Q-07", "Q-08", "Q-09",
                                           "Q-10", "Q-11", "Q-12", "Q-13", "Q-14", "Q-15", "Q-16", "Q-17", "Q-18"}


def _raw() -> dict:  # type: ignore[type-arg]
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_unknown_keys_rejected() -> None:
    raw = _raw()
    raw["sources"]["enigne"] = "postgres"
    with pytest.raises(ConfigError) as err:
        parse_config(raw)
    assert err.value.code == "CONFIG_INVALID"
    assert err.value.exit_code is ExitCode.USAGE


@pytest.mark.parametrize(("section", "key", "value"), [
    ("approvals", "deployment_approval_required", False),
    ("approvals", "metric_semantics_approval_required", False),
    ("approvals", "automatic_repair_limit", 4),
    ("production", "platform", "snowflake"),
])
def test_gate_and_platform_settings_cannot_be_weakened(section: str, key: str, value: object) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_unknown_marker_needs_reason() -> None:
    raw = _raw()
    raw["sources"]["engine"] = {"kind": "UNKNOWN", "reason": ""}
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_resolved_value_replaces_unknown() -> None:
    raw = _raw()
    raw["sources"]["engine"] = "Azure SQL Database"
    config = parse_config(raw)
    assert config.sources.engine == "Azure SQL Database"
    assert "sources.engine" not in {p for p, _, _ in unknown_fields(config)}


def test_allowlisted_env_overrides_apply() -> None:
    config = load_config(CONFIG, environ={"HARNESS_ARTIFACTS_DIR": "/tmp/a", "HARNESS_LOG_LEVEL": "DEBUG"})
    assert config.operational.artifacts_dir == "/tmp/a"
    assert config.operational.log_level == "DEBUG"
    assert set(ENV_OVERRIDES) == {"HARNESS_ARTIFACTS_DIR", "HARNESS_STATE_DIR", "HARNESS_LOG_LEVEL"}


def test_other_env_overrides_are_refused() -> None:
    with pytest.raises(ConfigError) as err:
        load_config(CONFIG, environ={"HARNESS_DEPLOYMENT_APPROVAL_REQUIRED": "false"})
    assert err.value.code == "CONFIG_ENV_OVERRIDE_FORBIDDEN"


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        load_config(tmp_path / "nope.yaml", environ={})
    assert err.value.code == "CONFIG_NOT_FOUND"
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [", encoding="utf-8")
    with pytest.raises(ConfigError) as err:
        load_config(bad, environ={})
    assert err.value.code == "CONFIG_NOT_YAML"


def test_json_schema_export_is_current() -> None:
    from harness.schemas import render_schemas

    for name, text in render_schemas().items():
        assert (Path("schemas") / name).read_text(encoding="utf-8") == text, f"schemas/{name} is stale"
