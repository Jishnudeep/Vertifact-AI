from datetime import datetime, timedelta, timezone

import pytest

from app.services.cache_policy import is_complete_run, is_fresh, unservable_reason

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
COMPLETE = {
    "factcheck": {"degraded": False},
    "crossref": {"degraded": False},
    "linguistic": {"degraded": False},
}


def _row(**overrides):
    row = {
        "confidence_level": "medium",
        "evidence_status": "corroborated",
        "agent_outputs": COMPLETE,
        "created_at": (NOW - timedelta(hours=1)).isoformat(),
    }
    row.update(overrides)
    return row


def test_servable_row_has_no_reason():
    assert unservable_reason(_row(), NOW) is None


@pytest.mark.parametrize("level", ["low", "insufficient", "LOW"])
def test_low_confidence_rows_are_not_served(level):
    assert "confidence" in unservable_reason(_row(confidence_level=level), NOW)


def test_legacy_rows_without_evidence_status_are_not_served():
    assert "evidence_status" in unservable_reason(_row(evidence_status=None), NOW)


def test_partial_runs_are_not_served():
    partial = {**COMPLETE, "crossref": {"degraded": True}}
    assert "partial" in unservable_reason(_row(agent_outputs=partial), NOW)


@pytest.mark.parametrize("agent_outputs", [None, {}, {"factcheck": {"degraded": False}}, "not-a-dict"])
def test_missing_agent_outputs_fail_closed(agent_outputs):
    assert is_complete_run(agent_outputs) is False


def test_missing_degraded_flag_fails_closed():
    outputs = {**COMPLETE, "linguistic": {}}
    assert is_complete_run(outputs) is False


@pytest.mark.parametrize("status,age,fresh", [
    ("fact_checked", timedelta(days=29), True),
    ("fact_checked", timedelta(days=31), False),
    ("corroborated", timedelta(hours=23), True),
    ("corroborated", timedelta(hours=25), False),
    ("disputed", timedelta(hours=25), False),
    ("unverified", timedelta(hours=25), False),
])
def test_ttl_depends_on_evidence_status(status, age, fresh):
    assert is_fresh(status, (NOW - age).isoformat(), NOW) is fresh


def test_supabase_z_suffix_and_naive_timestamps_parse():
    assert is_fresh("corroborated", "2026-09-23T11:00:00Z", NOW) is True
    assert is_fresh("corroborated", datetime(2026, 9, 23, 11, 0), NOW) is True  # naive -> UTC


@pytest.mark.parametrize("created_at", [None, "", "yesterday", 12345])
def test_unparseable_created_at_fails_closed(created_at):
    assert is_fresh("fact_checked", created_at, NOW) is False


def test_stale_rows_are_not_served():
    old = (NOW - timedelta(days=2)).isoformat()
    assert "stale" in unservable_reason(_row(created_at=old), NOW)
