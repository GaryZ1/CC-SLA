# metrics_pipeline_harness

A build harness that turns an Excel inventory of metric definitions into reviewed, deterministic
Databricks SQL pipelines (Bronze, Silver, dimensional Gold), with approval gates the coding agent
cannot bypass. The design is `../metrics_pipeline_harness_plan.html` (revision 2: D-01 Databricks SQL,
D-02 code is authoritative, D-03 sources in a cloud database).

**Status: phase P0 and the first half of P1** (see `docs/build_status.md`). Workbook ingestion works end
to end on the synthetic workbook; specs, planning, generation and execution are not built yet.
Everything under `fixtures/synthetic/` is invented and is not production evidence.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync --locked
```

## Commands

```bash
uv run harness --version
uv run harness config check                       # validate config/project.yaml, list every UNKNOWN
uv run python fixtures/synthetic/banking/make_workbook.py --verify
uv run harness profile fixtures/synthetic/banking/SYNTHETIC_retail_kpi_inventory.xlsx
uv run harness extract fixtures/synthetic/banking/SYNTHETIC_retail_kpi_inventory.xlsx --mapping config/column_mapping.yaml
uv run harness export-schemas --check
```

`extract` writes four artifacts to `artifacts/runs/<run_id>/` (also stored content-addressed under
`artifacts/sha256/`): `workbook_profile.json`, `raw_metric_rows.jsonl`, `ingestion_issues.csv` and the
`column_mapping.yaml` that was applied, plus `run_manifest.json` and a JSON-lines `run.log`. Identical
inputs give byte-identical artifacts.

Exit codes follow plan §12.2: 0 ok, 1 internal, 2 usage/config, 3 validation, 4 completed with blocked
items, 5 gate denied, 6 tests failed, 7 resource limit, 8 stale prerequisite, 9 transient target, 130 cancelled.

## Checks

```bash
uv run ruff check .
uv run mypy --strict src
uv run pytest
```

## Layout

| Path | Contents |
|---|---|
| `config/` | `project.yaml` (D-01..D-03, UNKNOWNs explicit), `ingest_limits.yaml`, `column_mapping.yaml` |
| `schemas/` | JSON Schemas exported from the Pydantic models (generated; drift-checked) |
| `specs/sources/` | Source contracts (SYNTHETIC) |
| `src/harness/ingest/` | Admission, bounded subprocess, two-pass reader, header mapping, classification, extraction |
| `src/harness/spec/` | Shared spec primitives and source contracts (MetricSpec and the expression AST next) |
| `fixtures/synthetic/banking/` | Synthetic workbook generator, CDC windows, variants, hand-calculated expectations |
| `tests/` | Unit, workbook (WB-xx) and fixture-independence tests |
| `docs/` | Configuration rules, open-question register, owner roster, build status and deviations |

## Safety properties already enforced

* Workbooks are untrusted: formulas are never evaluated, macros never run, links never refreshed; parsing
  runs in a subprocess with a wall-clock timeout and an address-space limit; zip bombs and oversize files
  are rejected before parsing.
* Hidden rows, columns and sheets are extracted and flagged, never dropped; merged cells are never
  forward-filled into business fields.
* Instruction-like text in cells, comments or hyperlinks is reported as an issue and has no other effect.
* Configuration cannot switch off approval gates, and the environment can override only paths and log level.
* Logs are JSON lines with secret redaction.
