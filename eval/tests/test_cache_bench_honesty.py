"""
Cache bench honesty (Task C review, B2): no fabricated baseline, no miss latencies passed off as hits,
seed rows that are servable and carry no invented bias label, scoped cleanup, and an explicit opt-in
before it touches a shared database.
"""
import json
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from app.config import settings
from eval import cache_bench
from eval.cache_bench import (
    BenchInputError,
    cleanup_bench_rows,
    load_live_baseline,
    run_cache_benchmark,
    seed_via_replay,
    summarize,
)

pytestmark = pytest.mark.anyio

GOOD_BASELINE = {"summary": {
    "run_status": "OK",
    "replay": {"mode": "record", "misses": 0},
    "latency": {"mode": "record", "p50_ms": 27500.0, "p95_ms": 33000.0, "n": 50},
    "external_calls_per_claim": {"mean": 8.1, "p50": 8.0, "max": 11},
}}


def _write(tmp_path, payload, name="record_run.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _variant(**changes):
    summary = json.loads(json.dumps(GOOD_BASELINE["summary"]))
    for key, value in changes.items():
        if value is None:
            summary.pop(key, None)
        else:
            summary[key] = value
    return {"summary": summary}


def _db_host():
    return urlparse(settings.supabase_url).hostname


def test_summarize_with_no_hits_reports_not_measured():
    res = summarize([], 27500.0, {"mean": 8.0})
    assert res["n_hits"] == 0
    assert res["hit_p50_ms"] is None and res["hit_p95_ms"] is None
    assert res["speedup"] is None and res["target_met"] is None


def test_baseline_is_read_with_provenance(tmp_path):
    path = _write(tmp_path, GOOD_BASELINE)
    baseline = load_live_baseline(path)
    assert baseline["live_p50_ms"] == 27500.0 and baseline["calls_per_claim"]["mean"] == 8.1
    assert baseline["source"] == str(path) and baseline["n"] == 50 and baseline["run_status"] == "OK"


@pytest.mark.parametrize("payload", [
    _variant(latency={"mode": "replay", "note": "not measured: replay timings"}),
    _variant(latency={"mode": "record-missing", "p50_ms": 900.0, "n": 3}),
    _variant(latency={"mode": "record", "p95_ms": 1.0, "n": 3}),
    _variant(latency=None),
    _variant(run_status="INVALID_RUN"),
    _variant(external_calls_per_claim=None),
    _variant(external_calls_per_claim={"mean": 0.0, "p50": 0.0, "max": 0}),
], ids=["replay-latency", "record-missing-latency", "no-p50", "no-latency", "invalid-run",
        "no-call-counts", "zero-call-counts"])
def test_baseline_that_cannot_support_the_comparison_is_refused(tmp_path, payload):
    with pytest.raises(BenchInputError):
        load_live_baseline(_write(tmp_path, payload))


def test_missing_baseline_file_is_refused_not_defaulted(tmp_path):
    with pytest.raises(BenchInputError):
        load_live_baseline(tmp_path / "does_not_exist.json")


PAIRS = [{"id": f"p{i}", "claim": f"Recorded claim {i}", "paraphrase": f"Claim {i} reworded"} for i in range(22)]
NEGATIVES = [{"id": "n0", "of": "p0", "text": "Recorded claim 0 negated"},
             {"id": "n1", "of": "p1", "text": "Recorded claim 1 with another number"}]


class FakeReplay:
    """Stands in for EvidenceReplay: counts install/uninstall and reports configured misses per claim."""

    def __init__(self, misses_for=()):
        self.misses_for = set(misses_for)
        self.installed = self.uninstalled = 0
        self.current = None

    def install(self):
        self.installed += 1

    def uninstall(self):
        self.uninstalled += 1

    def reset_claim_counts(self, claim_id=None):
        self.current = claim_id

    def claim_misses(self):
        return 1 if self.current in self.misses_for else 0


class FakeBench:
    """Stubbed embedder, pipeline, lookup and Supabase for run_cache_benchmark."""

    def __init__(self, hits_for=None, replay=None, verify_override=None):
        self.verified = []
        self.hits_for = hits_for if hits_for is not None else (lambda text, ids: None)
        self.replay = replay or FakeReplay()
        self.verify_override = verify_override
        self.supabase = MagicMock()
        table = self.supabase.table.return_value
        table.select.return_value.like.return_value.execute.return_value = MagicMock(data=[])
        table.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        table.delete.return_value.like.return_value.execute.return_value = MagicMock(data=[])

    def embed(self, text):
        return [0.1] * 384

    async def verify(self, text, bypass_cache=True):
        self.verified.append((text, bypass_cache))
        if self.verify_override:
            return await self.verify_override(text)
        return {"id": f"claim-{len(self.verified)}", "cached": False}

    async def lookup(self, embedding, text):
        return self.hits_for(text, [f"claim-{i + 1}" for i in range(len(self.verified))])

    def kwargs(self, tmp_path, pairs=PAIRS, negatives=NEGATIVES):
        return dict(record_summary_path=_write(tmp_path, GOOD_BASELINE), output_path=tmp_path / "out.json",
                    pairs=pairs, negatives=negatives, confirm_db=_db_host(), supabase=self.supabase,
                    embed=self.embed, verify=self.verify, replay=self.replay, lookup=self.lookup)


def _serves_all(text, ids):
    for i, pair in enumerate(PAIRS):
        if text in (pair["claim"], pair["paraphrase"]):
            return {"claim_id": ids[i]}
    return None


async def test_a_bench_where_the_cache_serves_nothing_refuses_and_reports_nothing(tmp_path):
    bench = FakeBench()
    with pytest.raises(BenchInputError, match="Only 0 of 22"):
        await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert not (tmp_path / "out.json").exists()
    assert bench.supabase.table.return_value.delete.called            # still cleaned up


async def test_a_hit_on_a_row_the_bench_did_not_seed_does_not_count(tmp_path):
    bench = FakeBench(hits_for=lambda text, ids: {"claim_id": "somebody-elses-row"})
    with pytest.raises(BenchInputError, match="Only 0 of 22"):
        await run_cache_benchmark(**bench.kwargs(tmp_path))


async def test_rows_are_seeded_by_the_real_pipeline_with_the_cache_enabled(tmp_path):
    bench = FakeBench(hits_for=_serves_all)
    await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert [t for t, _ in bench.verified] == [p["claim"] for p in PAIRS]
    assert all(bypass is False for _, bypass in bench.verified)
    assert bench.replay.installed == 1 and bench.replay.uninstalled == 1


async def test_hits_on_seeded_rows_are_timed_and_counted_with_provenance(tmp_path):
    report = await run_cache_benchmark(**FakeBench(hits_for=_serves_all).kwargs(tmp_path))
    n = len(PAIRS)
    assert report["pairs_served"] == n and report["pairs_not_served"] == []
    assert report["exact_lookups"]["hits"] == n and report["paraphrase_lookups"]["hits"] == n
    assert report["summary"]["n_hits"] == 2 * n and report["summary"]["target_met"] in (True, False)
    assert report["baseline"]["source"].endswith("record_run.json") and report["baseline"]["run_status"] == "OK"
    assert "recorded evidence" in report["seeding"]
    assert report["negative_lookups"]["false_positives"] == 0 and report["negative_lookups"]["passed"] is True


async def test_pairs_the_cache_does_not_serve_are_excluded_and_reported(tmp_path):
    unserved = {PAIRS[3]["claim"], PAIRS[7]["claim"]}

    def hits(text, ids):
        return None if text in unserved else _serves_all(text, ids)

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    assert report["pairs_served"] == len(PAIRS) - 2
    assert {r["id"] for r in report["pairs_not_served"]} == {"p3", "p7"}
    assert report["exact_lookups"]["n"] == len(PAIRS) - 2


async def test_negatives_of_unserved_pairs_are_not_probed(tmp_path):
    def hits(text, ids):
        return None if text == PAIRS[0]["claim"] else _serves_all(text, ids)

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    assert report["negatives_probed"] == 1 and report["negatives_skipped_unserved_pair"] == 1


async def test_negated_and_number_changed_claims_that_hit_are_false_positives(tmp_path):
    texts = {n["text"] for n in NEGATIVES}

    def hits(text, ids):
        return {"claim_id": ids[0]} if text in texts else _serves_all(text, ids)

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    assert report["negative_lookups"]["false_positives"] == len(NEGATIVES)
    assert report["negative_lookups"]["passed"] is False


async def test_any_replay_miss_aborts_the_bench_and_still_cleans_up(tmp_path):
    bench = FakeBench(hits_for=_serves_all, replay=FakeReplay(misses_for={PAIRS[4]["claim"]}))
    with pytest.raises(BenchInputError, match="Replay miss.*Recorded claim 4"):
        await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert bench.replay.uninstalled == 1
    assert bench.supabase.table.return_value.delete.called


async def test_a_claim_already_served_from_the_cache_is_not_a_seeded_row():
    async def cached(text):
        return {"id": "someone-elses", "cached": True}

    bench = FakeBench(verify_override=cached)
    res = await seed_via_replay(["a claim"], "abc12345", "unused", verify=bench.verify, replay=bench.replay)
    assert res["a claim"].claim_id is None and "already served" in res["a claim"].reason


async def test_a_claim_whose_agents_all_failed_is_reported_not_seeded():
    from app.agents.orchestrator import AllAgentsFailedError

    async def failed(text):
        raise AllAgentsFailedError("all failed")

    bench = FakeBench(verify_override=failed)
    res = await seed_via_replay(["a claim"], "abc12345", "unused", verify=bench.verify, replay=bench.replay)
    assert res["a claim"].claim_id is None and "every agent failed" in res["a claim"].reason


async def test_persisted_rows_are_tagged_without_touching_the_claim_or_embedding(monkeypatch):
    from app.agents import orchestrator

    recorded = []

    async def recorder(**kwargs):
        recorded.append(kwargs)
        return "claim-1"

    monkeypatch.setattr(orchestrator, "insert_cache", recorder)

    async def pipeline(text, bypass_cache=True):
        cid = await orchestrator.insert_cache(raw_input=text, extracted_claim=text, embedding=[0.2] * 4)
        return {"id": cid, "cached": False}

    replay = FakeReplay()
    await seed_via_replay(["Some claim"], "abc12345", "unused", verify=pipeline, replay=replay)
    assert recorded[0]["raw_input"] == "[bench:abc12345] Some claim"
    assert recorded[0]["extracted_claim"] == "Some claim" and recorded[0]["embedding"] == [0.2] * 4
    assert orchestrator.insert_cache is recorder, "insert_cache must be restored after seeding"


async def test_insert_cache_is_restored_and_replay_uninstalled_when_the_pipeline_blows_up(monkeypatch):
    from app.agents import orchestrator

    original = orchestrator.insert_cache

    async def boom(text, bypass_cache=True):
        raise RuntimeError("pipeline exploded")

    replay = FakeReplay()
    with pytest.raises(RuntimeError, match="pipeline exploded"):
        await seed_via_replay(["a claim"], "abc12345", "unused", verify=boom, replay=replay)
    assert orchestrator.insert_cache is original and replay.uninstalled == 1


async def test_refuses_to_touch_a_database_the_operator_did_not_name(tmp_path):
    bench = FakeBench()
    for wrong in (None, "some-other-project.supabase.co"):
        kwargs = bench.kwargs(tmp_path)
        kwargs["confirm_db"] = wrong
        with pytest.raises(BenchInputError, match="confirm"):
            await run_cache_benchmark(**kwargs)
    assert bench.verified == []


async def test_refuses_fewer_than_twenty_pairs_before_touching_anything(tmp_path):
    bench = FakeBench()
    with pytest.raises(BenchInputError, match="20"):
        await run_cache_benchmark(**bench.kwargs(tmp_path, pairs=PAIRS[:5]))
    assert bench.verified == []


async def test_cleanup_runs_when_a_lookup_blows_up_and_does_not_mask_the_error(tmp_path):
    bench = FakeBench()

    async def boom(embedding, text):
        raise RuntimeError("lookup exploded")

    bench.lookup = boom
    # the post-check also fails: the original error must still be the one raised
    bench.supabase.table.return_value.select.return_value.like.return_value.execute.return_value = MagicMock(
        data=[{"id": "x"}])
    with pytest.raises(RuntimeError, match="lookup exploded"):
        await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert bench.supabase.table.return_value.delete.called


async def test_a_paraphrase_that_matches_another_pairs_row_is_a_wrong_row_hit(tmp_path):
    def hits(text, ids):
        for i, pair in enumerate(PAIRS):
            if text == pair["claim"]:
                return {"claim_id": ids[i]}
            if text == pair["paraphrase"]:
                return {"claim_id": ids[(i + 1) % len(ids)]}        # somebody else's seeded row
        return None

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    assert report["paraphrase_lookups"]["wrong_row_hits"] == len(PAIRS)
    assert report["summary"]["n_hits"] == len(PAIRS)                  # only the exact claims counted


def test_cleanup_deletes_only_the_tagged_rows_and_the_given_ids():
    supabase = MagicMock()
    table = supabase.table.return_value
    table.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[{"id": "a"}])
    table.delete.return_value.like.return_value.execute.return_value = MagicMock(data=[])
    table.select.return_value.like.return_value.execute.return_value = MagicMock(data=[])
    cleanup_bench_rows(supabase, "abc12345", ["claim-1", "claim-2"])
    table.delete.return_value.like.assert_called_once_with("raw_input", "%[bench:abc12345]%")
    assert [c.args for c in table.delete.return_value.eq.call_args_list] == [("id", "claim-1"), ("id", "claim-2")]
    table.select.return_value.like.assert_called_once_with("raw_input", "%[bench:abc12345]%")


def test_sweep_removes_every_leftover_bench_row():
    supabase = MagicMock()
    table = supabase.table.return_value
    table.delete.return_value.like.return_value.execute.return_value = MagicMock(data=[{"id": "a"}, {"id": "b"}])
    assert cache_bench.sweep_bench_rows(supabase) == 2
    table.delete.return_value.like.assert_called_once_with("raw_input", "[bench:%")
