"""Project configuration model and loader (plan §1.3, backlog P0-02).

Rules:

* Unknown keys are rejected everywhere (``extra="forbid"``).
* ``UNKNOWN`` values are explicit ``{kind: UNKNOWN, reason, question}`` records and round-trip unchanged.
* Environment overrides are limited to an allowlist of *operational* settings (paths, log level).
  Nothing that affects gates, approvals, access allowlists, semantics or the target platform can be
  overridden from the environment; those change only through a reviewed edit of ``config/project.yaml``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

from harness.errors import ConfigError
from harness.spec.common import Frozen, Resolvable

SCHEMA_VERSION = "project_config/1.0.0"


class Mode(StrEnum):
    PLAN = "PLAN"
    BUILD = "BUILD"


class AuthorityPolicy(StrEnum):
    CODE = "CODE"
    ENGLISH = "ENGLISH"
    PER_METRIC_OWNER_DECISION = "PER_METRIC_OWNER_DECISION"


class Decision(Frozen):
    id: str = Field(pattern=r"^D-[0-9]{2}$")
    question: str = Field(pattern=r"^Q-[0-9]{2}[a-z]?$")
    answer: str
    interpretation: str


class Inputs(Frozen):
    workbook_paths: list[str]
    source_schemas: Resolvable[list[str]]
    data_dictionaries: Resolvable[list[str]]
    reference_code: Resolvable[list[str]]
    sample_data: Resolvable[list[str]]
    repository: Resolvable[str]


class LocalReference(Frozen):
    language: Literal["python"]
    engine: Literal["duckdb"]
    interface: Literal["cli"]
    data: Literal["synthetic", "synthetic_or_approved_anonymized"]


class LlmPolicy(Frozen):
    permitted_llm_data: Literal["none", "metadata_only_until_authorized", "metadata_only"]
    approved_provider_and_model: Resolvable[str]


class Approvals(Frozen):
    deployment_approval_required: Literal[True]
    metric_semantics_approval_required: Literal[True]
    automatic_repair_limit: int = Field(ge=0, le=3)


class Access(Frozen):
    allowed_read_sources: list[str]
    allowed_write_targets: list[str]


class Production(Frozen):
    platform: Literal["databricks"]
    sql_dialect: Literal["databricks"]
    unity_catalog: Resolvable[bool]
    cloud: Resolvable[str]
    compute: Resolvable[str]
    multi_statement_transactions: Resolvable[bool]
    dev_workspace: Resolvable[str]
    catalogs: Resolvable[dict[str, str]]
    orchestrator: Resolvable[str]


class Semantics(Frozen):
    authoritative_definition_policy: AuthorityPolicy
    excel_formula_rank: Resolvable[str]
    translation_rules_confirmed: Resolvable[bool]
    business_timezone: Resolvable[str]
    fiscal_calendar: Resolvable[str]
    period_bounds: Resolvable[str]
    incomplete_period_policy: Resolvable[str]
    base_currency: Resolvable[str]
    fx_policy: Resolvable[str]
    restatement_policy: Resolvable[str]
    tolerances: Resolvable[str]


class Sources(Frozen):
    location: Literal["cloud_database"]
    engine: Resolvable[str]
    access_path: Resolvable[str]
    change_capture: Resolvable[str]
    delete_representation: Resolvable[str]
    timestamp_storage: Resolvable[str]
    nonprod_copy_available: Resolvable[bool]
    owners: Resolvable[dict[str, str]]


class Workload(Frozen):
    metric_count: Resolvable[int]
    data_volumes: Resolvable[str]
    refresh_cadence: Resolvable[str]
    latency_target: Resolvable[str]
    history_required: Resolvable[str]


class Governance(Frozen):
    ci_platform: Resolvable[str]
    approvers: Resolvable[dict[str, list[str]]]
    gold_consumers: Resolvable[list[str]]
    existing_standards: Resolvable[str]
    retention: Resolvable[str]
    data_residency: Resolvable[str]
    classification: Resolvable[str]


class Operational(Frozen):
    """The only section the environment may override (see ``ENV_OVERRIDES``)."""

    artifacts_dir: str = "artifacts"
    state_dir: str = "state"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ProjectConfig(Frozen):
    schema_version: Literal["project_config/1.0.0"]
    project_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    mode: Mode
    decisions: list[Decision]
    inputs: Inputs
    local_reference: LocalReference
    llm: LlmPolicy
    approvals: Approvals
    access: Access
    production: Production
    semantics: Semantics
    sources: Sources
    workload: Workload
    governance: Governance
    operational: Operational = Operational()


# Environment variable -> field of ``operational``. Deliberately tiny; see module docstring.
ENV_OVERRIDES: Mapping[str, str] = {
    "HARNESS_ARTIFACTS_DIR": "artifacts_dir",
    "HARNESS_STATE_DIR": "state_dir",
    "HARNESS_LOG_LEVEL": "log_level",
}

DEFAULT_CONFIG_PATH = Path("config/project.yaml")


def parse_config(data: Any, *, source: str = "<memory>") -> ProjectConfig:
    try:
        return ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            "CONFIG_INVALID",
            f"{source}: {exc.error_count()} error(s)",
            evidence={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def load_config(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProjectConfig:
    """Load ``config/project.yaml`` and apply the allowlisted environment overrides."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("CONFIG_NOT_FOUND", str(path)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError("CONFIG_NOT_YAML", f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("CONFIG_INVALID", f"{path}: top level must be a mapping")
    env = os.environ if environ is None else environ
    overrides = {field: env[var] for var, field in ENV_OVERRIDES.items() if env.get(var)}
    rejected = sorted(
        var for var in env if var.startswith("HARNESS_") and var not in ENV_OVERRIDES and var != "HARNESS_CONFIG"
    )
    if rejected:
        raise ConfigError(
            "CONFIG_ENV_OVERRIDE_FORBIDDEN",
            "only operational settings may be overridden from the environment",
            evidence={"rejected": rejected, "allowed": sorted(ENV_OVERRIDES)},
        )
    if overrides:
        raw = dict(raw)
        raw["operational"] = {**(raw.get("operational") or {}), **overrides}
    return parse_config(raw, source=str(path))


def dump_config(config: ProjectConfig) -> str:
    """Canonical YAML: key order follows the model, which makes round trips stable."""
    data = config.model_dump(mode="json", exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def unknown_fields(config: ProjectConfig) -> list[tuple[str, str, str | None]]:
    """List every UNKNOWN in the config as (dotted path, reason, question id)."""
    found: list[tuple[str, str, str | None]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if value.get("kind") == "UNKNOWN":
                found.append((path, str(value.get("reason")), value.get("question")))
                return
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(config.model_dump(mode="json"), "")
    return found
