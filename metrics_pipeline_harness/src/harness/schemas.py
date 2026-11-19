"""JSON Schema export from the Pydantic models (plan §12.1 ``schemas/``: generated; diff reviewed).

``harness export-schemas`` writes them; ``harness export-schemas --check`` fails (exit 3) on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from harness.config import ProjectConfig
from harness.ingest.models import ColumnMapping, IngestConfig
from harness.spec.source import JoinContract, SourceContract

SCHEMAS: dict[str, type[BaseModel]] = {
    "project_config.schema.json": ProjectConfig,
    "ingest_limits.schema.json": IngestConfig,
    "column_mapping.schema.json": ColumnMapping,
    "source_contract.schema.json": SourceContract,
    "join_contract.schema.json": JoinContract,
}


def render_schemas() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, model in SCHEMAS.items():
        schema = model.model_json_schema(mode="validation")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"metrics_pipeline_harness/{name}"
        out[name] = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return out


def export_schemas(directory: Path, *, check: bool) -> list[str]:
    """Write (or with ``check``, compare) every schema; return the names that differ on disk."""
    stale: list[str] = []
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in render_schemas().items():
        target = directory / name
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != text:
            stale.append(name)
            if not check:
                target.write_text(text, encoding="utf-8")
    return stale
