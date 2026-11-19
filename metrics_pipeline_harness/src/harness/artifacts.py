"""Content-addressed artifact store and run directories (plan §2.1 "Artifacts").

Every artifact is stored once under ``<root>/sha256/<aa>/<hash>`` and referenced by hash. A run
directory ``<root>/runs/<run_id>/`` holds the human-readable copies plus ``run_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.errors import ArtifactWriteFailed

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_run_id(now_ms: int | None = None) -> str:
    """ULID-style identifier: 48-bit millisecond timestamp + 80 random bits, Crockford base32."""
    stamp = int(time.time() * 1000) if now_ms is None else now_ms
    value = (stamp << 80) | secrets.randbits(80)
    chars = []
    for _ in range(26):
        value, rem = divmod(value, 32)
        chars.append(_CROCKFORD[rem])
    return "".join(reversed(chars))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any, *, indent: int | None = 2) -> bytes:
    """Deterministic JSON: sorted keys at every level, UTF-8, trailing newline."""
    separators = None if indent else (",", ":")
    text = json.dumps(data, indent=indent, sort_keys=True, ensure_ascii=False, separators=separators)
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ArtifactWriteFailed("ARTIFACT_WRITE_FAILED", f"{path}: {exc}") from exc


@dataclass
class ArtifactStore:
    root: Path

    def put(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        target = self.root / "sha256" / digest[:2] / digest
        if not target.exists():
            _atomic_write(target, data)
        return digest

    def path_for(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest


@dataclass
class RunDir:
    store: ArtifactStore
    run_id: str
    command: str
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.store.root / "runs" / self.run_id

    def write(self, name: str, data: bytes) -> Path:
        digest = self.store.put(data)
        target = self.path / name
        _atomic_write(target, data)
        self.artifacts[name] = f"sha256:{digest}"
        return target

    def write_manifest(self, extra: dict[str, Any]) -> Path:
        manifest = {
            "schema_version": "run_manifest/1.0.0",
            "run_id": self.run_id,
            "command": self.command,
            "artifacts": dict(sorted(self.artifacts.items())),
            **extra,
        }
        target = self.path / "run_manifest.json"
        _atomic_write(target, canonical_json(manifest))
        return target
