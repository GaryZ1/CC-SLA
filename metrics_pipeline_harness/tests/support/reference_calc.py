"""Small pure-Python reference calculator over the SYNTHETIC fixture records (plan §11.2 independence rule).

It shares no code with the harness: it reads the fixture CSVs with the csv module and applies the
legacy code's semantics directly, so agreement with the hand-typed expectations is evidence that the
fixtures and the expectations describe the same thing.

Legacy semantics implemented (plan §5.2, §10):
* status comparisons are case-insensitive and ignore trailing spaces (TR-03)
* business date = date part of the stored wall-clock timestamp (TR-02, no conversion)
* numerator = COMPLETED; denominator = non-NULL status other than CANCELLED_BY_CUSTOMER
* product = the account's product version in effect at event time (SCD2, [from, to))
* CDC: latest source_seq per txn_id; an exact re-read is a no-op; the same (txn_id, seq) with different
  content is a sequence conflict and both versions are excluded
* end-of-day balance = last non-NULL snapshot of the day as stored (pandas ``.last()``), no carry-forward
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, localcontext
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic" / "banking"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _legacy_eq(value: str | None, literal: str) -> bool | None:
    if value is None:
        return None
    return value.rstrip().upper() == literal.rstrip().upper()


@dataclass(frozen=True)
class Txn:
    txn_id: str
    account_id: str
    status: str | None
    amount: Decimal
    currency: str
    event_ts: datetime
    source_seq: int
    op: str


def load_txn_changes(windows: list[str], variant: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for window in windows:
        rows.extend(_rows(FIXTURES / "windows" / window / "core_txn_changes.csv"))
    if variant:
        vdir = FIXTURES / "variants" / variant
        extra = _rows(vdir / "core_txn_changes.csv")
        if "operation: replace" in (vdir / "variant.yaml").read_text(encoding="utf-8"):
            keys = {(r["txn_id"], r["source_seq"]) for r in extra}
            rows = [r for r in rows if (r["txn_id"], r["source_seq"]) not in keys]
        rows.extend(extra)
    return rows


def apply_cdc(rows: list[dict[str, str]]) -> tuple[dict[str, Txn], set[tuple[str, int]]]:
    versions: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        versions[(row["txn_id"], int(row["source_seq"]))].append(row)
    conflicts = {key for key, group in versions.items() if len({tuple(sorted(r.items())) for r in group}) > 1}
    latest: dict[str, Txn] = {}
    for (txn_id, seq), group in sorted(versions.items(), key=lambda item: item[0][1]):
        if (txn_id, seq) in conflicts:
            continue
        row = group[0]
        txn = Txn(
            txn_id=txn_id,
            account_id=row["account_id"],
            status=row["status"] if row["status"] != "" else None,
            amount=Decimal(row["amount"]),
            currency=row["currency"],
            event_ts=datetime.fromisoformat(row["event_ts"]),
            source_seq=seq,
            op=row["op"],
        )
        if txn.op == "D":
            latest.pop(txn_id, None)
        else:
            latest[txn_id] = txn
    return latest, conflicts


def product_at(account_id: str, at: datetime) -> str:
    history = sorted(
        (datetime.fromisoformat(r["valid_from"]), r["product_code"])
        for r in _rows(FIXTURES / "windows" / "window_1" / "core_account_product_hist.csv")
        if r["account_id"] == account_id
    )
    current = None
    for valid_from, product in history:
        if valid_from <= at:
            current = product
    if current is None:
        raise LookupError(f"no product version for {account_id} at {at}")
    return current


@dataclass(frozen=True)
class DailyCell:
    business_date: date
    account_id: str
    product_code: str
    transaction_count: int
    transaction_amount_cad: Decimal
    completion_numerator: int
    completion_denominator: int

    @property
    def rate(self) -> Decimal | None:
        return ratio(self.completion_numerator, self.completion_denominator)

    @property
    def status(self) -> str:
        return "OK" if self.completion_denominator else "NO_ELIGIBLE_ATTEMPTS"


def ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    with localcontext() as ctx:
        ctx.prec = 50
        return Decimal(numerator) / Decimal(denominator)


def txn_daily(windows: list[str], variant: str | None = None) -> list[DailyCell]:
    latest, _ = apply_cdc(load_txn_changes(windows, variant))
    acc: dict[tuple[date, str, str], list[Txn]] = defaultdict(list)
    for txn in latest.values():
        acc[(txn.event_ts.date(), txn.account_id, product_at(txn.account_id, txn.event_ts))].append(txn)
    cells = []
    for (business_date, account_id, product_code), txns in sorted(acc.items()):
        completed = [t for t in txns if _legacy_eq(t.status, "COMPLETED")]
        eligible = [t for t in txns if _legacy_eq(t.status, "CANCELLED_BY_CUSTOMER") is False]
        cells.append(DailyCell(
            business_date, account_id, product_code,
            transaction_count=len(completed),
            transaction_amount_cad=sum((t.amount for t in completed), Decimal("0.00")),
            completion_numerator=len(completed),
            completion_denominator=len(eligible),
        ))
    return cells


def balance_daily() -> list[tuple[date, str, str, Decimal]]:
    rows = _rows(FIXTURES / "windows" / "window_1" / "core_balance_snapshot.csv")
    by_day: dict[tuple[str, date], list[tuple[datetime, int, Decimal | None]]] = defaultdict(list)
    for row in rows:
        ts = datetime.fromisoformat(row["as_of_ts"])
        value = Decimal(row["ledger_balance"]) if row["ledger_balance"] else None
        by_day[(row["account_id"], ts.date())].append((ts, int(row["source_seq"]), value))
    out = []
    for (account_id, day), snaps in sorted(by_day.items(), key=lambda item: (item[0][1], item[0][0])):
        non_null = [s for s in sorted(snaps) if s[2] is not None]
        if not non_null:
            continue
        day_end = datetime.combine(day, datetime.max.time())
        value = non_null[-1][2]
        assert value is not None
        out.append((day, account_id, product_at(account_id, day_end), value))
    return out
