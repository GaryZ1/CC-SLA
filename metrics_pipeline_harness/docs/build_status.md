# Build status against the phased backlog (plan §13)

Status as of this commit. "Done" means implemented with tests passing locally; it does not mean the
backlog item's external acceptance (CI green on the org's CI, second-person review, named owners) is met.

| Item | Status | Notes |
|---|---|---|
| P0-01 Repository skeleton, lockfile, CI lint + test (no deploy) | Done | `uv.lock`; `harness --version`; workflow `.github/workflows/metrics-harness-ci.yml` at the repo root. CI vendor unknown (Q-08), GitHub Actions used as a stand-in. |
| P0-02 Project config model and loader | Done | UNKNOWNs round-trip byte-identically; unknown keys rejected; env override rules in `docs/configuration.md`; `schemas/project_config.schema.json`. |
| P0-03 Open-question register and owner roster | Partial | `docs/decisions/open_questions.md`, `docs/owners.yaml`. Named people and due dates are UNKNOWN. |
| P0-04 Synthetic contracts, fixtures, hand-calculated expectations | Done, review pending | `specs/sources/srcdb_core_*.yaml`, `fixtures/synthetic/banking/**`, deterministic `make_workbook.py --verify`. E1-E4 cross-checked by an independent pure-Python calculator; second-person review still required. |
| P0-05 Real workbook sample through an approved data path | Not started | External; depends on Q-07, Q-13. |
| P1-01 Admission, bounds, bounded subprocess | Done | WB-10 (corrupt, zip bomb, limits), WB-12, WB-13; timeout and RLIMIT_AS memory limit tested. |
| P1-02 Two-pass reader and discovery | Done | WB-01..WB-04, WB-08; profile matches §4.5. |
| P1-03 Header detection and column mapping | Done | Multirow, merged and repeated headers; unmapped columns preserved under `extra`. |
| P1-04 Metric ID registry | Not started | Next. Duplicate business IDs are already reported at extraction (WB-07). |
| P1-05 Pydantic IR, expression AST, type checker, JSON Schema export | Started | Shared `Unknown`/`NotApplicable`/`Resolvable` primitives and `SourceContract`/`JoinContract`; MetricSpec and the AST are next. |
| P1-06 .. P1-10, P2+ | Not started | |

## Deviations from the plan, and why

* **Location.** The harness lives in `metrics_pipeline_harness/` inside the existing CC-SLA repository
  (which holds unrelated notebooks), rather than as a repository root. Q-12 is still open.
* **Python 3.12** pinned in `.python-version` (plan allows 3.11 or 3.12); all dependencies are pinned in `uv.lock`.
* **Issue ISS-0004 cells.** The plan's example cites `A3` (the row); the harness reports a missing column
  once per sheet at the header row (`A2:D2`) instead of once per row, to avoid one issue per row on large sheets.
* **Extra issues beyond ISS-0001..0005** on the synthetic workbook: ISS-0006/0007 (empty definition and
  code on hidden draft row 10) and ISS-0008 (unmapped `Notes` column on `Legacy`). The plan's five expected
  issues keep their IDs because issues are numbered by stage (structure, untrusted text, mapping, rows, notes).
* **`TYPED_VALUE` cell class** added for numbers, dates and booleans typed into a cell (not formulas);
  the plan's six classes have no slot for them.
* **`missing_severity` / `empty_severity`** options added to `column_mapping.yaml` so the optional
  `business_id` and `code` fields can raise the MINOR issues the plan shows (ISS-0004, ISS-0005).
* **Workbook hash.** The plan's synthetic hash `3a8f1c0e…` is illustrative; the generated fixture's real
  hash is recorded in `fixtures/synthetic/banking/SYNTHETIC_retail_kpi_inventory.xlsx.sha256`.
* **Zip entries are stored uncompressed** in the synthetic workbook so its bytes do not depend on the zlib build.
* **Derived variant expectation.** `variants.csv` adds MET-0003 = 1/2 for `seq_conflict` (not in §10.6):
  with both T002 versions quarantined, the denominator is T001 + T003. Marked as derived; needs review.
