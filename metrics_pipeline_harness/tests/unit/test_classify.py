"""Stage 9-10 classification and language detection."""

from __future__ import annotations

from harness.ingest.classify import Language, detect_language, looks_like_python, sql_parse_dialects

DIALECTS = ["tsql", "postgres", "mysql", "oracle", "snowflake", "databricks", "duckdb"]


def test_sql_statement_detected_without_choosing_dialect() -> None:
    d = detect_language("SELECT COUNT(*) FROM core.txn WHERE status = 'COMPLETED'", DIALECTS)
    assert d.detected is Language.SQL
    assert d.dialect is None
    assert d.confidence == 0.55
    assert d.evidence[0].startswith("parses: tsql")


def test_tsql_markers_are_a_hint_only() -> None:
    d = detect_language("SELECT COUNT(DISTINCT account_id) FROM core.txn WHERE txn_date >= DATEADD(day, -90, "
                        "GETDATE())", DIALECTS)
    assert d.detected is Language.SQL and d.dialect == "tsql" and d.confidence == 0.75
    assert any("GETDATE" in e for e in d.evidence)


def test_python_detected_by_parsing_not_execution() -> None:
    code = "df.sort_values('as_of_ts').groupby(['account_id', df.as_of_ts.dt.date]).ledger_balance.last()"
    assert sql_parse_dialects(code, DIALECTS) == []      # bare expressions are not SQL statements
    d = detect_language(code, DIALECTS)
    assert d.detected is Language.PYTHON


def test_canary_python_is_never_executed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    canary = tmp_path / "canary"
    code = f"open({str(canary)!r}, 'w').write('ran')"
    assert detect_language(code, DIALECTS).detected is Language.PYTHON
    assert not canary.exists()                               # SG-12


def test_english_and_identifiers_are_not_code() -> None:
    for text in ("count of completed transactions", "TBD", "RB-KPI-001 / RB-KPI-005", "Retail Payments Analytics"):
        assert not sql_parse_dialects(text, DIALECTS)
        assert not looks_like_python(text)
        assert detect_language(text, DIALECTS).detected is Language.UNKNOWN


def test_dax_and_sas_lexical_markers() -> None:
    assert detect_language("Total := CALCULATE(SUM('Txn'[amount]))", DIALECTS).detected is Language.DAX
    assert detect_language("proc sql;\n  select * from x;\nquit;\nrun;", DIALECTS).detected is Language.SAS


def test_declared_language_raises_confidence() -> None:
    assert detect_language("SELECT 1 FROM t", DIALECTS, declared="SQL").confidence == 0.7
