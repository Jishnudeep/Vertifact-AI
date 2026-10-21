"""
Cache bench honesty (Task C review, B2): no fabricated baseline, no miss latencies passed off as hits,
seed rows that are servable and carry no invented bias label, scoped cleanup, and an explicit opt-in
before it touches a shared database.
"""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from app.config import settings
from app.services.bias_lookup import get_bias_label
from app.services.cache_policy import unservable_reason
from eval import cache_bench
from eval.cache_bench import (
    BENCH_NEGATIVES,
    BENCH_PAIRS,
    BenchInputError,
    cleanup_bench_rows,
    load_live_baseline,
    run_cache_benchmark,
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


class FakeBench:
    """Stubbed embedder, inserter, lookup and Supabase for run_cache_benchmark."""

    def __init__(self, hits_for=None):
        self.inserted = []
        self.hits_for = hits_for if hits_for is not None else (lambda text, ids: None)
        self.supabase = MagicMock()
        table = self.supabase.table.return_value
        table.select.return_value.like.return_value.execute.return_value = MagicMock(data=[])
        table.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
        table.delete.return_value.like.return_value.execute.return_value = MagicMock(data=[])

    def embed(self, text):
        return [0.1] * 384

    async def insert(self, **kwargs):
        self.inserted.append(kwargs)
        return f"claim-{len(self.inserted)}"

    async def lookup(self, embedding, text):
        return self.hits_for(text, [f"claim-{i + 1}" for i in range(len(self.inserted))])

    def kwargs(self, tmp_path):
        return dict(record_summary_path=_write(tmp_path, GOOD_BASELINE), output_path=tmp_path / "out.json",
                    confirm_db=_db_host(), supabase=self.supabase, embed=self.embed, insert=self.insert,
                    lookup=self.lookup)


async def test_all_misses_never_report_miss_latencies_as_hits(tmp_path):
    report = await run_cache_benchmark(**FakeBench().kwargs(tmp_path))
    assert report["summary"]["n_hits"] == 0
    assert report["summary"]["target_met"] is None and report["summary"]["hit_p95_ms"] is None


async def test_a_hit_on_a_row_the_bench_did_not_seed_does_not_count(tmp_path):
    bench = FakeBench(hits_for=lambda text, ids: {"claim_id": "somebody-elses-row"})
    report = await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert report["summary"]["n_hits"] == 0
    assert report["exact_lookups"]["hits"] == 0


async def test_hits_on_seeded_rows_are_timed_and_counted_with_provenance(tmp_path):
    def hits(text, ids):
        for i, pair in enumerate(BENCH_PAIRS):
            if text in (pair["claim"], pair["paraphrase"]):
                return {"claim_id": ids[i]}
        return None

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    n = len(BENCH_PAIRS)
    assert report["exact_lookups"]["hits"] == n and report["paraphrase_lookups"]["hits"] == n
    assert report["summary"]["n_hits"] == 2 * n and report["summary"]["target_met"] in (True, False)
    assert report["baseline"]["source"].endswith("record_run.json") and report["baseline"]["run_status"] == "OK"
    assert report["negative_lookups"]["false_positives"] == 0 and report["negative_lookups"]["passed"] is True


async def test_negated_and_number_changed_claims_that_hit_are_false_positives(tmp_path):
    negatives = {n["text"] for n in BENCH_NEGATIVES}
    bench = FakeBench(hits_for=lambda text, ids: {"claim_id": ids[0]} if text in negatives else None)
    report = await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert report["negative_lookups"]["false_positives"] == len(BENCH_NEGATIVES)
    assert report["negative_lookups"]["passed"] is False


async def test_seeded_rows_are_servable_complete_runs_with_no_invented_bias_labels(tmp_path):
    bench = FakeBench()
    await run_cache_benchmark(**bench.kwargs(tmp_path))
    assert bench.inserted, "the bench must seed rows"
    for kwargs in bench.inserted:
        row = {**kwargs, "created_at": datetime.now(timezone.utc).isoformat()}
        assert unservable_reason(row, datetime.now(timezone.utc)) is None
        assert kwargs["raw_input"].startswith("[bench:")
        for src in kwargs["source_citations"]:
            assert src["bias_label"] == get_bias_label(src["domain"])
        assert any("benchmark" in w.lower() for w in kwargs["warnings"])


async def test_refuses_to_touch_a_database_the_operator_did_not_name(tmp_path):
    bench = FakeBench()
    for wrong in (None, "some-other-project.supabase.co"):
        kwargs = bench.kwargs(tmp_path)
        kwargs["confirm_db"] = wrong
        with pytest.raises(BenchInputError, match="confirm"):
            await run_cache_benchmark(**kwargs)
    assert bench.inserted == []


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


async def test_seed_rows_claim_no_certainty(tmp_path):
    bench = FakeBench()
    await run_cache_benchmark(**bench.kwargs(tmp_path))
    for kwargs in bench.inserted:
        assert kwargs["evidence_status"] == "unverified"
        assert kwargs["c_total"] == 0.5 and kwargs["s_fact"] == 0.5 and kwargs["v_consensus"] == 0.0
        assert kwargs["source_citations"] == []


async def test_a_paraphrase_that_matches_another_pairs_row_is_a_wrong_row_hit(tmp_path):
    def hits(text, ids):
        pair = next((i for i, p in enumerate(BENCH_PAIRS) if text in (p["claim"], p["paraphrase"])), None)
        if pair is None:
            return None
        return {"claim_id": ids[(pair + 1) % len(ids)]}      # always somebody else's seeded row

    report = await run_cache_benchmark(**FakeBench(hits_for=hits).kwargs(tmp_path))
    assert report["summary"]["n_hits"] == 0
    assert report["exact_lookups"]["wrong_row_hits"] == len(BENCH_PAIRS)
    assert report["seeded_rows"] == len(BENCH_PAIRS)
