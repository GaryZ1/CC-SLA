# Configuration

| File | Model | Purpose |
|---|---|---|
| `config/project.yaml` | `harness.config.ProjectConfig` | Plan §1.3 configuration with decisions D-01..D-03; every unknown is an explicit `UNKNOWN` record |
| `config/ingest_limits.yaml` | `harness.ingest.models.IngestConfig` | Plan §4.3 bounds and inclusion rules |
| `config/column_mapping.yaml` | `harness.ingest.models.ColumnMapping` | Plan §4.4 header aliases and sheet overrides |
| `specs/sources/*.yaml` | `harness.spec.source.SourceContract` | Plan §4.10 source contracts (SYNTHETIC for now) |

JSON Schemas for all of them are generated into `schemas/` (`uv run harness export-schemas`; CI runs
`--check` and fails on drift).

## Rules

* **Unknown keys are rejected** in every file. A typo is an error, never a silently ignored setting.
* **Three kinds of missing.** `{kind: UNKNOWN, reason, question}` (evidence not yet available),
  `{kind: NOT_APPLICABLE, justification}` (concept does not apply), and business `null` on fields declared
  nullable. Loading and dumping `config/project.yaml` is byte-identical, so UNKNOWNs round-trip unchanged.
* **Gates cannot be weakened by configuration.** `deployment_approval_required` and
  `metric_semantics_approval_required` only accept `true`; `automatic_repair_limit` is capped at 3; the
  production platform and dialect only accept `databricks` (D-01).

## Environment overrides

Only operational settings may come from the environment:

| Variable | Field |
|---|---|
| `HARNESS_ARTIFACTS_DIR` | `operational.artifacts_dir` |
| `HARNESS_STATE_DIR` | `operational.state_dir` |
| `HARNESS_LOG_LEVEL` | `operational.log_level` |

Any other `HARNESS_*` variable makes the CLI exit 2 (`CONFIG_ENV_OVERRIDE_FORBIDDEN`). Approvals, access
allowlists, semantics and the target platform change only through a reviewed edit of `config/project.yaml`.
Secrets are never read from config files; `.env.example` lists secret *names* only.
