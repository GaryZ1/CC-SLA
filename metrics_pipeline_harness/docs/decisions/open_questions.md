# Open-question register

Source: plan §1.5. Answered decisions are recorded in `config/project.yaml` (`decisions:`); every
open question also appears as an `UNKNOWN` with its `question:` id there (`uv run harness config check`).
**Person** and **Due** are not yet assigned (P0-03 acceptance needs both).

| ID | Status | Question | Blocks | Continues meanwhile | Owner role | Person | Due |
|---|---|---|---|---|---|---|---|
| Q-01 | ANSWERED (D-01) | Production platform and dialect: Databricks, SQL | — | — | — | — | — |
| Q-01a | OPEN | Unity Catalog? Cloud? Serverless/pro SQL warehouse or job compute? Multi-statement transactions? Dev workspace/catalog for the harness? | Production adapter, controller credentials, P2-08, P4 | Design assumes Unity Catalog, no multi-statement transactions | Platform admin | UNKNOWN | UNKNOWN |
| Q-02 | ANSWERED (D-02) | Authority policy: code wins | — | — | — | — | — |
| Q-02a | OPEN | Confirm translation rules TR-01..TR-05 once; where do Excel formula cells rank (A-10)? | Metrics using those constructs (MET-0005 in the example) | All other metrics | Metrics governance lead | UNKNOWN | UNKNOWN |
| Q-03 | OPEN | Business timezone, fiscal calendar, period bounds, incomplete-period policy | Every time-bucketed metric | Non-temporal validation, design | Finance / metric owners | UNKNOWN | UNKNOWN |
| Q-04 | OPEN | Base currency, FX source, rate type, conversion timing | Multi-currency monetary metrics | Single-currency metrics | Finance | UNKNOWN | UNKNOWN |
| Q-05 | OPEN | Restatement policy; per-metric tolerances | Correction handling; any release needing a tolerance | Design with both options parameterized | Metric owners | UNKNOWN | UNKNOWN |
| Q-06 | PARTLY ANSWERED (D-03) | Sources are in a cloud database; owners per source still needed | All real mappings | Proposed contracts, synthetic slice | Data owners | UNKNOWN | UNKNOWN |
| Q-07 | OPEN | Is an LLM approved? Provider, model, classification, residency | LLM-assisted proposals only | Deterministic pipeline and human review | Security / data protection | UNKNOWN | UNKNOWN |
| Q-08 | OPEN | CI/CD platform and plan, repository visibility, administrator | Final release-gate implementation | Controller design, gate test suite | DevOps / platform admin | UNKNOWN | UNKNOWN |
| Q-09 | OPEN | Approvers (metric owners, data owners, architect, release reviewers) and how they authenticate | Any real approval | Synthetic approvals with a local test key | Program sponsor | UNKNOWN | UNKNOWN |
| Q-10 | OPEN | Metric count, volumes, cadence, latency target, history | Performance design, SLAs | Functional correctness | Product owner | UNKNOWN | UNKNOWN |
| Q-11 | OPEN | Gold consumers and interface | Publication interface | Gold views as default contract | Analytics lead | UNKNOWN | UNKNOWN |
| Q-12 | OPEN | Existing repository, pipeline framework or coding standard? | Repository layout finalization | Greenfield layout (§12) | Engineering lead | UNKNOWN | UNKNOWN |
| Q-13 | OPEN | Retention/deletion obligations; data residency | Bronze retention and purge | Retention as configuration | Legal / privacy | UNKNOWN | UNKNOWN |
| Q-14 | OPEN | Source database engine/product; is it the engine the workbook code targets? | Translation of code-defined metrics; extraction connector | Dialect detection from code samples; synthetic slice | Data owner, DBA | UNKNOWN | UNKNOWN |
| Q-15 | OPEN | Federation, landing pipeline or managed connector? CDC/change tracking/updated-at? Delete representation? | Bronze and Silver for real sources | Extraction design with all options | DBA, platform admin | UNKNOWN | UNKNOWN |
| Q-16 | OPEN | Timestamp storage: UTC or local wall time, with or without offset? | Time-bucketed metrics from real sources | Ingest naive timestamps unchanged as TIMESTAMP_NTZ | DBA | UNKNOWN | UNKNOWN |
| Q-17 | OPEN | Non-production copy of the source engine for read-only legacy queries on fixtures? | Legacy-reference reconciliation | Hand-calculated expectations | DBA, security | UNKNOWN | UNKNOWN |
| Q-18 | OPEN | Databricks Jobs, or an existing orchestrator that triggers jobs? | Production scheduling | Databricks Jobs assumed (A-12) | Platform admin | UNKNOWN | UNKNOWN |
