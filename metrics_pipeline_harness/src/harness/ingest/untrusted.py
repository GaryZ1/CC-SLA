"""Stage 11: flag instruction-like text in workbook content (plan §4.1, WB-11).

Flagging is informational. Workbook text, comments and hyperlinks are data; nothing found here can
change policy, readiness or approvals, and no code path reads these findings as instructions.
"""

from __future__ import annotations

import re

PATTERNS: tuple[tuple[str, str], ...] = (
    ("override_instructions",
     r"\b(ignore|disregard|forget|override)\b.{0,40}\b(rules?|instructions?|polic(y|ies)|prompts?|guidelines)\b"),
    ("approval_directive", r"\b(mark|set|flag|treat|consider)\b.{0,40}\b(approved|verified|ready|signed[- ]off)\b"),
    ("approval_directive", r"\b(auto[- ]?approve|approve (all|every|everything))\b"),
    ("gate_bypass", r"\b(bypass|skip|disable)\b.{0,30}\b(gates?|approvals?|reviews?|checks?|validation)\b"),
    ("role_injection", r"\b(you are now|act as|system prompt|developer message|assistant:)\b"),
    ("tool_directive", r"\b(run|execute|call|invoke)\b.{0,25}\b(tool|command|shell|script|function|macro)\b"),
    ("url_directive", r"\b(fetch|open|visit|download|curl|wget|browse to)\b.{0,30}(https?://|www\.)"),
)

_COMPILED = tuple((name, re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)) for name, pattern in PATTERNS)


def scan_text(text: str) -> list[str]:
    """Return the sorted names of matching patterns (empty list when nothing looks like an instruction)."""
    if not text:
        return []
    return sorted({name for name, regex in _COMPILED if regex.search(text)})


def excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
