from copy import deepcopy

import pytest

from so101.aws_budget import GIB, Reservations, estimate


def contract():
    return {
        "schema": 1, "spec_sha256": "a" * 64,
        "temporary_copy_policy_sha256": "b" * 64,
        "estimate": {
            "bytes_retained": GIB, "retention_hours": 24, "upload_bytes": GIB,
            "download_bytes": GIB, "requests": 10,
            "storage_micro_per_gib_hour": 100,
            "transfer_micro_per_gib": 100_000, "request_micro": 10,
            "fixed_micro": 0,
        },
    }


def test_estimate_rounds_up_and_counts_both_transfer_directions():
    fields = contract()["estimate"]
    assert estimate(**fields) == 202_500
    fields.update(bytes_retained=1, upload_bytes=1, download_bytes=0, requests=0)
    assert estimate(**fields) == 2


@pytest.mark.parametrize("key,value", [
    ("bytes_retained", True), ("requests", -1), ("retention_hours", 0),
    ("transfer_micro_per_gib", 0), ("fixed_micro", 0.1),
])
def test_invalid_or_undeclared_costs_fail_closed(key, value):
    fields = contract()["estimate"]
    fields[key] = value
    with pytest.raises(ValueError):
        estimate(**fields)


def test_reservation_persists_and_replay_does_not_release_allowance(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = Reservations(path)
    assert ledger.reserve("1" * 32, contract()) == 202_500
    ledger.close()
    ledger = Reservations(path)
    try:
        assert ledger.reserve("1" * 32, contract()) == 202_500
        changed = deepcopy(contract())
        changed["estimate"]["fixed_micro"] = 1
        with pytest.raises(ValueError, match="different"):
            ledger.reserve("1" * 32, changed)
        for number in range(2, 25):
            ledger.reserve(f"{number:032x}", contract())
        with pytest.raises(ValueError, match="exhausted"):
            ledger.reserve("f" * 32, contract())
    finally:
        ledger.close()


def test_exact_limit_and_independent_connections(tmp_path):
    path = tmp_path / "budget.sqlite"
    first, second = Reservations(path), Reservations(path)
    try:
        large = contract()
        large["estimate"]["fixed_micro"] = 5_000_000 - 202_500
        assert first.reserve("1" * 32, large) == 5_000_000
        with pytest.raises(ValueError, match="exhausted"):
            second.reserve("2" * 32, contract())
    finally:
        first.close()
        second.close()
