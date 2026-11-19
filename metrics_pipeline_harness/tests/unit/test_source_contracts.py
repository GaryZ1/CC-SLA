"""P0-04 / P1-05 start: synthetic source contracts validate; malformed ones are rejected."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from harness.spec.common import EvidenceStatus, Unknown
from harness.spec.source import SourceContract, load_source_contracts, validate_logical_type

SPECS = Path("specs/sources")


def test_all_synthetic_contracts_validate() -> None:
    contracts = load_source_contracts(SPECS)
    assert sorted(contracts) == [
        "SRCDB.core.account_product_hist", "SRCDB.core.balance_snapshot", "SRCDB.core.product", "SRCDB.core.txn",
    ]
    for contract in contracts.values():
        assert contract.evidence_status is EvidenceStatus.SYNTHETIC
        assert isinstance(contract.engine, Unknown) and contract.engine.question == "Q-14"
    txn = contracts["SRCDB.core.txn"]
    assert txn.column("event_ts").logical_type == "TIMESTAMP_NTZ"     # ADR-009: naive timestamps unchanged
    assert txn.column("amount").logical_type == "DECIMAL(18,2)"


def test_contract_columns_match_fixture_headers() -> None:
    for contract in load_source_contracts(SPECS).values():
        assert contract.fixture
        for fixture in sorted(Path().glob(contract.fixture)):
            header = fixture.read_text(encoding="utf-8").splitlines()[0].split(",")
            assert header == [c.name for c in contract.columns if c.name in header], fixture
            assert set(header) <= {c.name for c in contract.columns}, fixture


def _txn() -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((SPECS / "srcdb_core_txn.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutate", [
    lambda d: d["columns"][0].update(logical_type="VARCHAR"),
    lambda d: d.update(business_keys=["nope"]),
    lambda d: d["columns"][0].update(nullable=True),           # business key must be NOT NULL
    lambda d: d.update(event_timestamp="missing_col"),
    lambda d: d.update(evidence_status="CATALOG_VERIFIED"),    # fixture paths only on SYNTHETIC
    lambda d: d.update(unexpected_key=1),
])
def test_invalid_contracts_rejected(mutate) -> None:  # type: ignore[no-untyped-def]
    data = _txn()
    mutate(data)
    with pytest.raises(ValidationError):
        SourceContract.model_validate(data)


def test_logical_types() -> None:
    assert validate_logical_type("DECIMAL(38, 12)") == "DECIMAL(38,12)"
    for bad in ("DECIMAL(39,2)", "DECIMAL(5,6)", "STRING"):
        with pytest.raises(ValueError):
            validate_logical_type(bad)
