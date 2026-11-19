"""Developer CLI (plan §12.2).

Every command logs JSON lines to stderr and to ``artifacts/runs/<run_id>/run.log``, accepts ``--config``
and prints the run ID. There is deliberately no deploy command: deployment is requested from the
protected controller, which verifies approvals and holds the credentials.

Implemented so far: ``profile``, ``extract``, ``config check``, ``export-schemas``. Other §12.2 commands
arrive with their backlog items (validate-specs: P1-07, plan: P2-01, generate: P2-02, run/test: P2-04/P2-05, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness import __version__
from harness.artifacts import ArtifactStore, RunDir, new_run_id
from harness.config import DEFAULT_CONFIG_PATH, ProjectConfig, dump_config, load_config, unknown_fields
from harness.errors import ExitCode, HarnessError
from harness.log import get_logger, setup_logging

DEFAULT_LIMITS = Path("config/ingest_limits.yaml")
DEFAULT_MAPPING = Path("config/column_mapping.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="metrics_pipeline_harness developer CLI")
    parser.add_argument("--version", action="version", version=f"harness {__version__}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="project config (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="profile a workbook (writes workbook_profile.json)")
    profile.add_argument("workbook", type=Path)
    profile.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    profile.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)

    extract = sub.add_parser("extract", help="extract metric rows, issues and profile from a workbook")
    extract.add_argument("workbook", type=Path)
    extract.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    extract.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)

    schemas = sub.add_parser("export-schemas", help="write JSON Schemas exported from the models")
    schemas.add_argument("--dir", type=Path, default=Path("schemas"))
    schemas.add_argument("--check", action="store_true", help="exit 3 if any schema on disk is stale")

    config = sub.add_parser("config", help="project configuration commands")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    check = config_sub.add_parser("check", help="validate config/project.yaml and list UNKNOWN values")
    check.add_argument("--print", dest="print_config", action="store_true", help="print the canonical config")
    return parser


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _ingest(args: argparse.Namespace, config: ProjectConfig, run: RunDir, *, profile_only: bool) -> ExitCode:
    from harness.ingest.extract import ingest_workbook
    from harness.ingest.models import load_column_mapping, load_ingest_config

    log = get_logger()
    ingest_config = load_ingest_config(args.limits)
    mapping_bytes = args.mapping.read_bytes() if args.mapping.exists() else b""
    mapping = load_column_mapping(args.mapping)
    result = ingest_workbook(
        args.workbook, config=ingest_config, mapping=mapping, mapping_bytes=mapping_bytes,
        display_path=_display_path(args.workbook),
    )
    run.write("workbook_profile.json", result.profile_bytes())
    if not profile_only:
        run.write("raw_metric_rows.jsonl", result.records_bytes())
        run.write("ingestion_issues.csv", result.issues_bytes())
        run.write("column_mapping.yaml", mapping_bytes)
    counts = result.profile["issue_counts"]
    log.info("ingest_completed", extra={
        "workbook_sha256": result.admission.sha256, "records": len(result.records),
        "sheets_extracted": result.sheets_extracted, "issue_counts": counts,
    })
    run.write_manifest({
        "inputs": {
            "workbook": {"path": _display_path(args.workbook), "sha256": result.admission.sha256},
            "mapping": _display_path(args.mapping),
            "limits": _display_path(args.limits),
        },
        "summary": {"records": len(result.records), "issue_counts": counts},
    })
    return ExitCode.COMPLETED_WITH_BLOCKED if result.blocker_count else ExitCode.OK


def _config_check(args: argparse.Namespace, config: ProjectConfig) -> ExitCode:
    unknowns = unknown_fields(config)
    if args.print_config:
        sys.stdout.write(dump_config(config))
    else:
        summary: dict[str, Any] = {
            "project_name": config.project_name,
            "mode": config.mode.value,
            "platform": config.production.platform,
            "authority_policy": config.semantics.authoritative_definition_policy.value,
            "unknown_count": len(unknowns),
            "unknown": [{"path": p, "question": q, "reason": r} for p, r, q in unknowns],
        }
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return ExitCode.OK


def _export_schemas(args: argparse.Namespace) -> ExitCode:
    from harness.schemas import export_schemas

    stale = export_schemas(args.dir, check=args.check)
    get_logger().info("schemas_exported", extra={"stale": stale, "check": args.check})
    if args.check and stale:
        print(f"stale schemas: {', '.join(stale)}", file=sys.stderr)
        return ExitCode.VALIDATION_FAILED
    return ExitCode.OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    run_id = new_run_id()
    started = time.monotonic()
    command = args.command if args.command != "config" else f"config {args.config_command}"
    log = setup_logging(context={"run_id": run_id, "command": command})
    try:
        config = load_config(args.config)
        store = ArtifactStore(Path(config.operational.artifacts_dir))
        run = RunDir(store, run_id, command)
        log = setup_logging(config.operational.log_level, run_log=run.path / "run.log" if command in (
            "profile", "extract") else None, context={"run_id": run_id, "command": command})
        log.info("run_started", extra={"harness_version": __version__})
        print(f"run_id={run_id}")
        if args.command == "profile":
            code = _ingest(args, config, run, profile_only=True)
        elif args.command == "extract":
            code = _ingest(args, config, run, profile_only=False)
        elif command == "config check":
            code = _config_check(args, config)
        elif args.command == "export-schemas":
            code = _export_schemas(args)
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"unknown command {args.command}")
        if command in ("profile", "extract"):
            print(f"artifacts={run.path}")
    except HarnessError as exc:
        log.error("run_failed", extra={"error_type": type(exc).__name__, "code": exc.code, "detail": exc.message,
                                       "evidence": exc.evidence, "exit_code": int(exc.exit_code)})
        print(f"error {exc.code}: {exc.message}", file=sys.stderr)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        log.error("run_cancelled")
        return int(ExitCode.CANCELLED)
    except Exception:
        log.exception("internal_error")
        return int(ExitCode.INTERNAL)
    log.info("run_finished", extra={"exit_code": int(code), "duration_ms": int((time.monotonic() - started) * 1000)})
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
