"""Log redaction (part of SG-14: secrets absent from logs)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from harness.log import redact, setup_logging

FAKE_SECRETS = [
    "password=hunter2-fake",
    "client_secret: abcDEF123fake",
    "Authorization: Bearer eyJfakefakefake.payload.sig",
    "dapi" + "0" * 32,
    "AKIA" + "FAKEFAKEFAKEFAKE",
]


def test_redact_removes_planted_fake_secrets() -> None:
    for secret in FAKE_SECRETS:
        text, count = redact(f"connecting with {secret} now")
        assert count >= 1
        assert "hunter2" not in text and "abcDEF123" not in text and "0" * 32 not in text
        assert "FAKEFAKEFAKEFAKE" not in text and "eyJfake" not in text


def test_log_lines_are_json_and_redacted(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    log_file = tmp_path / "run.log"
    logger = setup_logging("INFO", run_log=log_file, context={"run_id": "R1"})
    logger.info("connect", extra={"dsn": "host=x password=hunter2-fake", "rows": 3})
    for handler in logger.handlers:
        handler.flush()
    line = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert line["event"] == "connect" and line["run_id"] == "R1" and line["rows"] == 3
    assert "hunter2" not in json.dumps(line)
    assert line["redactions_applied"] == 1
    assert "hunter2" not in capsys.readouterr().err
    logging.getLogger("harness").handlers.clear()
