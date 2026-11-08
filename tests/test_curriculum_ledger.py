import pytest

from so101.curriculum_ledger import CurriculumLedger


def pair(index, category="separated"):
    return {
        "status": "complete", "scene_family_id": f"family-{index}",
        "split": "development", "tags": {"category": category},
        "successful_episodes": [f"/pair-{index}/0", f"/pair-{index}/1"],
        "attempts": [
            {"example_id": f"{index}:{i}", "source": f"/pair-{index}/{i}", "outcome": "success"}
            for i in range(2)
        ],
    }


def test_family_membership_is_reserved_and_never_reassigned(tmp_path):
    ledger = CurriculumLedger(tmp_path / "ledger.sqlite", audit_seed=7)
    with pytest.raises(ValueError, match="reserved"):
        ledger.accept_pair(pair(0))
    ledger.reserve_family("family-0", "development")
    with pytest.raises(ValueError, match="different split"):
        ledger.reserve_family("family-0", "train")
    ledger.close()


def test_category_audits_and_replay_survive_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = CurriculumLedger(path, audit_seed=7)
    for category in ("separated", "pile"):
        for index in range(10):
            report = pair(f"{category}-{index}", category)
            ledger.reserve_family(report["scene_family_id"], "development")
            first = ledger.accept_pair(report)
            assert ledger.accept_pair(report) == first
    pending = ledger.pending_audits()
    assert len(pending) == 4
    assert [row["category"] for row in pending].count("pile") == 2
    ledger.close()
    ledger = CurriculumLedger(path, audit_seed=7)
    assert ledger.pending_audits() == pending
    ledger.record_audit(pending[0]["example_id"], passed=False, findings="Target occluded")
    assert len(ledger.pending_audits()) == 3
    ledger.close()
    with pytest.raises(ValueError, match="seed changed"):
        CurriculumLedger(path, audit_seed=8)


def test_incomplete_pair_does_not_increment_audit_population(tmp_path):
    ledger = CurriculumLedger(tmp_path / "ledger.sqlite", audit_seed=7)
    report = pair(0)
    ledger.reserve_family(report["scene_family_id"], "development")
    report["attempts"][1]["outcome"] = "failed"
    with pytest.raises(ValueError, match="complete"):
        ledger.accept_pair(report)
    assert ledger.connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0] == 0
    ledger.close()


def test_global_sequences_remain_distinct_across_workers_and_restarts(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = CurriculumLedger(path, audit_seed=7)
    second = CurriculumLedger(path, audit_seed=7)
    assert first.claim_sequence(minimum=15) == 15
    assert second.claim_sequence(minimum=15) == 16
    assert first.claim_sequence(minimum=16) == 17
    first.close()
    second.close()
    resumed = CurriculumLedger(path, audit_seed=7)
    assert resumed.claim_sequence(minimum=0) == 18
    resumed.close()
