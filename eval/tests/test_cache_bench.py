"""
Tests for cache benchmark (Plan 9, Task 6).
Offline unit tests for summarize() and cleanup_bench_rows().
"""
from unittest.mock import MagicMock
import pytest

from eval.cache_bench import BenchInputError, cleanup_bench_rows, load_pairs, summarize


def _pairs_file(tmp_path, n=20, negatives=None, **overrides):
    import json
    pairs = [{"id": f"p{i}", "claim": f"Recorded claim number {i} about a thing", "paraphrase": f"Claim {i} reworded"}
             for i in range(n)]
    data = {"pairs": pairs, "negatives": negatives if negatives is not None else
            [{"id": "n0", "of": "p0", "text": "Recorded claim number 0 about nothing"}]}
    data.update(overrides)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_pairs_accepts_a_valid_file(tmp_path):
    pairs, negatives = load_pairs(_pairs_file(tmp_path))
    assert len(pairs) == 20 and negatives[0]["of"] == "p0"


def test_load_pairs_needs_at_least_twenty_pairs(tmp_path):
    with pytest.raises(BenchInputError, match="20"):
        load_pairs(_pairs_file(tmp_path, n=19))


def test_load_pairs_rejects_a_paraphrase_identical_to_the_claim(tmp_path):
    import json
    path = _pairs_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pairs"][3]["paraphrase"] = data["pairs"][3]["claim"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BenchInputError, match="p3"):
        load_pairs(path)


def test_load_pairs_rejects_a_negative_of_an_unknown_pair(tmp_path):
    path = _pairs_file(tmp_path, negatives=[{"id": "n0", "of": "nope", "text": "x"}])
    with pytest.raises(BenchInputError, match="n0"):
        load_pairs(path)


def test_summarize_target_met():
    # 100 hit measurements from 10ms to 50ms
    hit_ms = [10.0 + i * 0.4 for i in range(100)]
    live_p50 = 4000.0  # 10% is 400ms; hit_p95 will be ~48ms, well under 400ms
    calls = {"mean": 3.45, "p50": 3.0, "max": 5}

    res = summarize(hit_ms, live_p50, calls)

    assert res["n_hits"] == 100
    assert 29.0 <= res["hit_p50_ms"] <= 31.0
    assert res["hit_p95_ms"] < 400.0
    assert res["target_met"] is True
    assert res["speedup"] > 100.0
    assert res["calls_saved_per_hit"] == 3.45


def test_summarize_target_missed():
    # Hit p95 is 500ms; live_p50 is 2000ms -> 10% is 200ms; 500ms is not < 200ms
    hit_ms = [100.0, 300.0, 500.0]
    live_p50 = 2000.0
    calls = {"mean": 2.5}

    res = summarize(hit_ms, live_p50, calls)

    assert res["target_met"] is False
    assert res["hit_p95_ms"] > 200.0


def test_summarize_edge_cases():
    # Empty
    empty_res = summarize([], 1000.0, {})
    assert empty_res["n_hits"] == 0
    assert empty_res["target_met"] is None
    assert empty_res["speedup"] is None

    # Single item
    single_res = summarize([50.0], 1000.0, {"mean": 3.0})
    assert single_res["n_hits"] == 1
    assert single_res["hit_p50_ms"] == 50.0
    assert single_res["hit_p95_ms"] == 50.0
    assert single_res["target_met"] is True  # 50 < 100


def test_cleanup_bench_rows_success():
    supabase = MagicMock()
    table_mock = MagicMock()
    supabase.table.return_value = table_mock

    # mock delete().eq().execute() and delete().like().execute()
    del_mock = MagicMock()
    table_mock.delete.return_value = del_mock
    del_mock.eq.return_value.execute.return_value = MagicMock(data=[{"id": "claim-1"}])
    del_mock.like.return_value.execute.return_value = MagicMock(data=[{"id": "claim-2"}])

    # mock select().like().execute() -> returns 0 remaining rows
    sel_mock = MagicMock()
    table_mock.select.return_value = sel_mock
    sel_mock.like.return_value.execute.return_value = MagicMock(data=[])

    deleted = cleanup_bench_rows(supabase, "test1234", ["claim-1"])
    assert deleted == 2


def test_cleanup_bench_rows_raises_if_rows_remain():
    supabase = MagicMock()
    table_mock = MagicMock()
    supabase.table.return_value = table_mock

    del_mock = MagicMock()
    table_mock.delete.return_value = del_mock
    del_mock.eq.return_value.execute.return_value = MagicMock(data=[])
    del_mock.like.return_value.execute.return_value = MagicMock(data=[])

    # mock select().like().execute() -> returns 1 remaining row (failed to clean up!)
    sel_mock = MagicMock()
    table_mock.select.return_value = sel_mock
    sel_mock.like.return_value.execute.return_value = MagicMock(data=[{"id": "dangling-1"}])

    with pytest.raises(RuntimeError, match="Cleanup failed"):
        cleanup_bench_rows(supabase, "test1234", ["claim-1"])
