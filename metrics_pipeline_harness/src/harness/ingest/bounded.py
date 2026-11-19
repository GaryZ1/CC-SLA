"""Resource-bounded execution of untrusted-input parsing (plan §4.7).

``run_bounded`` runs a module-level function in a fresh ``spawn`` subprocess, applies an address-space
limit (RLIMIT_AS on Linux; the container's cgroup limit is the outer bound in CI), kills it on timeout
and returns its JSON-serializable result. Failures come back as structured ``HarnessError`` values.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.errors import HarnessError, ResourceLimitExceeded, WorkbookCorrupt, error_from_dict


def _apply_memory_limit(memory_mb: int) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    import resource

    limit = memory_mb * 1024 * 1024
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return True


def _child(
    fn: Callable[..., Any], args: tuple[Any, ...], memory_mb: int, out_path: str
) -> None:  # pragma: no cover - runs in the subprocess
    payload: dict[str, Any]
    try:
        _apply_memory_limit(memory_mb)
        payload = {"ok": True, "result": fn(*args)}
    except HarnessError as exc:
        payload = {"ok": False, "error": exc.as_dict()}
    except MemoryError:
        payload = {"ok": False, "error": ResourceLimitExceeded("WB_MEMORY_LIMIT", f"> {memory_mb} MB").as_dict()}
    except Exception as exc:
        payload = {
            "ok": False,
            "error": WorkbookCorrupt(
                "WB_PARSE_FAILED",
                f"{type(exc).__name__}: {exc}",
                evidence={"trace": traceback.format_exc(limit=8)},
            ).as_dict(),
        }
    try:
        Path(out_path).write_text(json.dumps(payload, default=str), encoding="utf-8")
    except MemoryError:
        Path(out_path).write_text(
            json.dumps({"ok": False, "error": ResourceLimitExceeded("WB_MEMORY_LIMIT", "result").as_dict()}),
            encoding="utf-8",
        )


def run_bounded(fn: Callable[..., Any], *args: Any, timeout_s: float, memory_mb: int) -> Any:
    """Run ``fn(*args)`` in a bounded subprocess and return its JSON-decoded result."""
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="harness-bounded-") as tmp:
        out_path = str(Path(tmp) / "result.json")
        proc = ctx.Process(target=_child, args=(fn, args, memory_mb, out_path), daemon=True)
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
            raise ResourceLimitExceeded("WB_TIMEOUT", f"parse exceeded {timeout_s}s wall clock")
        out = Path(out_path)
        if not out.exists():
            # Killed by the OS (for example the OOM killer) or crashed before writing a result.
            detail = f"parse subprocess exited with {proc.exitcode} and no result"
            if proc.exitcode in (-9, 137):
                raise ResourceLimitExceeded("WB_MEMORY_LIMIT", detail)
            raise WorkbookCorrupt("WB_PARSE_FAILED", detail)
        payload = json.loads(out.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise error_from_dict(payload["error"])
    return payload["result"]
