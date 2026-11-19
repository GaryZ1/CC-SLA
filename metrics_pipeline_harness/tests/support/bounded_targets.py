"""Module-level functions for bounded-subprocess tests (the spawn start method needs importable targets)."""

from __future__ import annotations

import time

from harness.errors import ResourceLimitExceeded


def add(a: int, b: int) -> int:
    return a + b


def sleep_for(seconds: float) -> str:
    time.sleep(seconds)
    return "woke"


def allocate(megabytes: int) -> int:
    block = bytearray(megabytes * 1024 * 1024)
    return len(block)


def raise_structured() -> None:
    raise ResourceLimitExceeded("WB_SHEET_LIMIT", "from child")


def raise_plain() -> None:
    raise RuntimeError("boom")
