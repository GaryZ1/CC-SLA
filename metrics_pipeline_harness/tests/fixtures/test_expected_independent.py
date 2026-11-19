"""P0-04: the hand-typed expectations (plan §10.6) agree with an independent reference calculation."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from tests.support import reference_calc as ref

EXPECTED = ref.FIXTURES / "expected"
TOL = Decimal("1E-17")


def _csv(name: str) -> list[dict[str, str]]:
    with (EXPECTED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(got: Decimal | None, expected: str) -> bool:
    if expected == "":
        return got is None
    return got is not None and abs(got - Decimal(expected)) <= TOL


def test_e1_matches_reference() -> None:
    expected = _csv("e1_txn_daily_window1.csv")
    got = ref.txn_daily(["window_1"])
    assert len(got) == len(expected) == 5
    for cell, row in zip(got, expected, strict=True):
        assert (str(cell.business_date), cell.account_id, cell.product_code) == (
            row["business_date"], row["account_id"], row["product_code"])
        assert cell.transaction_count == int(row["transaction_count"])
        assert cell.transaction_amount_cad == Decimal(row["transaction_amount_cad"])
        assert (cell.completion_numerator, cell.completion_denominator) == (
            int(row["completion_numerator"]), int(row["completion_denominator"]))
        assert _close(cell.rate, row["payment_completion_rate"])
        assert cell.status == row["completion_rate_status"]


def _rollup(cells: list[ref.DailyCell], key) -> dict[str, tuple[int, Decimal, int, int]]:  # type: ignore[no-untyped-def]
    out: dict[str, list] = {}
    for c in cells:
        agg = out.setdefault(key(c), [0, Decimal("0.00"), 0, 0])
        agg[0] += c.transaction_count
        agg[1] += c.transaction_amount_cad
        agg[2] += c.completion_numerator
        agg[3] += c.completion_denominator
    return {k: (v[0], v[1], v[2], v[3]) for k, v in out.items()}


def test_e2_rollups_recompute_from_components() -> None:
    cells = ref.txn_daily(["window_1"])
    grains = {
        "code_grain": lambda c: f"{c.business_date}|{c.account_id}",
        "business_date": lambda c: str(c.business_date),
        "product": lambda c: c.product_code,
        "total": lambda c: "all",
    }
    rolled = {g: _rollup(cells, fn) for g, fn in grains.items()}
    for row in _csv("e2_rollups_window1.csv"):
        count, amount, num, den = rolled[row["grain"]][row["group_key"]]
        assert count == int(row["transaction_count"])
        assert amount == Decimal(row["transaction_amount_cad"])
        assert (num, den) == (int(row["completion_numerator"]), int(row["completion_denominator"]))
        assert _close(ref.ratio(num, den), row["payment_completion_rate"])


def test_e2_average_of_ratios_trap_differs() -> None:
    rates = [c.rate for c in ref.txn_daily(["window_1"]) if c.rate is not None]
    average = sum(rates, Decimal(0)) / len(rates)
    assert abs(average - Decimal("0.8333333333333333")) < Decimal("1E-15")
    assert abs(average - Decimal(5) / Decimal(7)) > Decimal("0.1")


def test_e3_balance_follows_code_without_carry_forward() -> None:
    expected = _csv("e3_balance_daily_window1.csv")
    got = ref.balance_daily()
    assert [(str(d), a, p, v) for d, a, p, v in got] == [
        (r["balance_date"], r["account_id"], r["product_code"], Decimal(r["eod_ledger_balance_cad"]))
        for r in expected
    ]
    questions = {r["question"]: r["result"] for r in _csv("e3_balance_questions.csv")}
    for day in ("2026-03-07", "2026-03-08", "2026-03-09"):
        total = sum((v for d, _, _, v in got if str(d) == day), Decimal("0.00"))
        assert total == Decimal(questions[f"total_balance_{day}"])
    assert sum((v for *_, v in got), Decimal("0.00")) == Decimal(questions["sum_over_days_2026-03-07..09"])


def test_e4_correction_restates_only_its_date() -> None:
    before = {(c.business_date, c.account_id): c for c in ref.txn_daily(["window_1"])}
    after = {(c.business_date, c.account_id): c for c in ref.txn_daily(["window_1", "window_2"])}
    e4 = {r["assertion"]: r["expected"] for r in _csv("e4_after_window2.csv")}
    changed = {k for k in after if after[k] != before.get(k)}
    assert {int(d.strftime("%Y%m%d")) for d, _ in changed} == {int(e4["fact_transaction_rebuilt_date_keys"])}
    cell = after[(date(2026, 3, 8), "000456")]
    assert cell.transaction_amount_cad == Decimal(e4["amount_2026-03-08_000456_SAV-HIGH"])
    assert cell.transaction_count == int(e4["count_2026-03-08_000456_SAV-HIGH"])
    day = [c for k, c in after.items() if k[0] == date(2026, 3, 8)]
    assert sum((c.transaction_amount_cad for c in day), Decimal(0)) == Decimal(e4["total_amount_2026-03-08"])
    total_amount = sum((c.transaction_amount_cad for c in after.values()), Decimal(0))
    assert total_amount == Decimal(e4["period_total_amount"])
    num = sum(c.completion_numerator for c in after.values())
    den = sum(c.completion_denominator for c in after.values())
    assert num == int(e4["period_total_count"])
    assert _close(ref.ratio(num, den), e4["period_rate"])


def test_window_2_reread_is_not_a_conflict() -> None:
    latest, conflicts = ref.apply_cdc(ref.load_txn_changes(["window_1", "window_2"]))
    assert conflicts == set()
    assert latest["T001"].source_seq == 1
    assert latest["T005"].source_seq == 9


@pytest.mark.parametrize("row", _csv("variants.csv"), ids=lambda r: f"{r['variant']}-{r['metric']}")
def test_variants(row: dict[str, str]) -> None:
    cells = {(str(c.business_date), c.account_id): c for c in ref.txn_daily(["window_1"], variant=row["variant"])}
    expected = row["expected"]
    if expected == "EXCLUDED":
        currencies = {t.currency for t in ref.apply_cdc(ref.load_txn_changes(["window_1"], row["variant"]))[0].values()}
        assert currencies - {"CAD"}, "guard DQ-CCY-01 must have something to fail on"
        return
    cell = cells[(row["business_date"], row["account_id"])]
    if row["metric"] == "MET-0001":
        assert cell.transaction_count == int(expected)
    elif row["metric"] == "MET-0002":
        assert cell.transaction_amount_cad == Decimal(expected)
    elif row["metric"] == "MET-0003":
        fraction, value = (part.strip() for part in expected.split("="))
        num, den = (int(x) for x in fraction.split("/"))
        assert (cell.completion_numerator, cell.completion_denominator) == (num, den)
        assert _close(cell.rate, value)


def test_every_fixture_file_is_labelled_synthetic() -> None:
    for path in sorted(ref.FIXTURES.rglob("*")):
        if path.suffix in {".yaml", ".md", ".py"} and path.is_file():
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".yaml" and "variant:" in text:
                continue   # variant descriptors live under the SYNTHETIC fixture tree and say what they test
            assert "SYNTHETIC" in text, f"{path} must say SYNTHETIC"
    assert Path(ref.FIXTURES / "SYNTHETIC_retail_kpi_inventory.xlsx").exists()
