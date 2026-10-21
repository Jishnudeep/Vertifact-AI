"""
Cache benchmark with counted call savings (Plan 9, Task 6 / Gate D4).

Measures semantic cache hit latency (embedding + pgvector lookup + NLI equivalence, the same path
verify_claim takes) for exact claims and faithful paraphrases, and checks that negated or
number-changed variants MISS. The comparison baseline (live p50, calls per claim) comes only from a
valid record run; there are no fallback constants, and a figure that was not measured is None.

Seed rows are produced by the real pipeline: verify_claim(bypass_cache=False) under EvidenceReplay in
replay mode, on claims the record run already recorded. So every seeded row is exactly what the scoring
engine produced from real recorded evidence (nothing synthetic, no invented bias labels, no quota). Any
replay miss aborts the bench. A row counts only if the cache then actually serves it for its exact claim.

It writes tagged rows into the configured Supabase project, so it refuses to run unless the operator
names that project (--confirm-db <host>), and it always cleans up in `finally`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional
from urllib.parse import urlparse

# Ensure repo root and backend are on sys.path (the script is run as `python eval/cache_bench.py`)
BASE_DIR = Path(__file__).resolve().parent.parent
for _p in (BASE_DIR, BASE_DIR / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.config import settings  # noqa: E402

logger = logging.getLogger("cache_bench")

DEFAULT_OUT = BASE_DIR / "eval" / "results" / "cache_bench.json"
DEFAULT_RECORD_SUMMARY = BASE_DIR / "eval" / "results" / "record_run.json"
DEFAULT_EVIDENCE_CACHE = BASE_DIR / "eval" / "results" / "evidence_cache"
MIN_SERVED_PAIRS = 20           # plan Task 6: N >= 20 rows the cache really serves

TAG_PREFIX = "[bench:"
# Latency from these modes is a live measurement; a replay or record-missing run mixes cached turns in.
LIVE_LATENCY_MODES = {"record", "live"}


class BenchInputError(RuntimeError):
    """The bench cannot produce an honest figure (or may not touch this database); nothing was run."""


def load_pairs(path: Path | str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Bench pairs from a JSON file: {"pairs": [{"id", "claim", "paraphrase"}], "negatives": [{"id", "of", "text"}]}.
    Each pair's `claim` must be the exact text of a claim the record run recorded (the bench replays it
    to seed the row); a paraphrase is a faithful rewording, a negative a negated / number-changed
    variant of pair `of` that must MISS. Negatives and paraphrases are never run through the pipeline.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs, negatives = data.get("pairs") or [], data.get("negatives") or []
    ids = [p.get("id") for p in pairs]
    if len(pairs) < MIN_SERVED_PAIRS or len(set(ids)) != len(ids):
        raise BenchInputError(f"{path}: need >= {MIN_SERVED_PAIRS} pairs with unique ids, got {len(pairs)}.")
    for p in pairs:
        if not (p.get("claim") and p.get("paraphrase")) or p["claim"] == p["paraphrase"]:
            raise BenchInputError(f"{path}: pair {p.get('id')!r} needs a claim and a different paraphrase.")
    for n in negatives:
        if n.get("of") not in ids or not n.get("text"):
            raise BenchInputError(f"{path}: negative {n.get('id')!r} must name an existing pair in `of` and carry text.")
    return pairs, negatives


def summarize(
    hit_ms: List[float],
    live_p50_ms: Optional[float],
    calls_per_claim: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summary of cache hit latencies against the live p50. Anything that was not measured is None:
    no hits means no latency, no speedup and no verdict on the target.
    target_met: hit_p95 < 0.10 * live_p50; calls_saved_per_hit: calls_per_claim.mean.
    """
    n = len(hit_ms)
    if n >= 2:
        q = statistics.quantiles(hit_ms, n=100, method="inclusive")
        hit_p50, hit_p95 = round(q[49], 2), round(q[94], 2)
    elif n == 1:
        hit_p50 = hit_p95 = round(hit_ms[0], 2)
    else:
        hit_p50 = hit_p95 = None

    have_both = n > 0 and live_p50_ms is not None and live_p50_ms > 0
    return {
        "n_hits": n,
        "hit_p50_ms": hit_p50,
        "hit_p95_ms": hit_p95,
        "live_p50_ms": live_p50_ms,
        "speedup": round(live_p50_ms / hit_p50, 2) if have_both and hit_p50 else None,
        "target_met": bool(hit_p95 < 0.10 * live_p50_ms) if have_both else None,
        "calls_saved_per_hit": float(calls_per_claim["mean"]) if calls_per_claim.get("mean") else None,
    }


def load_live_baseline(path: Path | str) -> Dict[str, Any]:
    """
    Live p50 and calls per claim from a valid record run, with their provenance.
    Refuses (BenchInputError) rather than substituting a default: a replay, record-missing or
    invalid run, or one without call counts, cannot support the comparison.
    """
    path = Path(path)
    if not path.exists():
        raise BenchInputError(f"No record run at {path}. Run the record run first; the bench has no fallback.")
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", data)

    if summary.get("run_status") != "OK":
        raise BenchInputError(f"{path} has run_status={summary.get('run_status')!r}; need a valid (OK) run.")
    latency = summary.get("latency") or {}
    if latency.get("mode") not in LIVE_LATENCY_MODES:
        raise BenchInputError(
            f"{path} latency.mode={latency.get('mode')!r}: replay and record-missing timings are not live latency.")
    if not latency.get("p50_ms"):
        raise BenchInputError(f"{path} has no latency.p50_ms.")
    calls = summary.get("external_calls_per_claim") or {}
    if not calls.get("mean"):
        raise BenchInputError(f"{path} has no counted external_calls_per_claim.mean (live mode does not count calls).")

    return {
        "source": str(path),
        "timestamp": summary.get("timestamp") or data.get("timestamp"),
        "run_status": summary["run_status"],
        "n": latency.get("n"),
        "live_p50_ms": float(latency["p50_ms"]),
        "calls_per_claim": calls,
    }


def _tag(bench_run_id: str) -> str:
    return f"{TAG_PREFIX}{bench_run_id}]"


def cleanup_bench_rows(supabase: Any, bench_run_id: str, claim_ids: Optional[List[str]] = None) -> int:
    """
    Delete the rows inserted for this benchmark run from claims (cascades to results), by id and by
    tag, then assert zero tagged rows remain. Scoped strictly to this run's tag.
    """
    deleted_count = 0
    pattern = f"%{_tag(bench_run_id)}%"

    for cid in claim_ids or []:
        try:
            res = supabase.table("claims").delete().eq("id", cid).execute()
            deleted_count += len(res.data or [])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error deleting claim {cid}: {e}")

    try:
        res2 = supabase.table("claims").delete().like("raw_input", pattern).execute()
        deleted_count += len(res2.data or [])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Error deleting claims by tag {pattern}: {e}")

    check = supabase.table("claims").select("id").like("raw_input", pattern).execute()
    remaining = len(check.data or [])
    if remaining > 0:
        raise RuntimeError(f"Cleanup failed: {remaining} rows with tag '{pattern}' still remain in Supabase claims table.")
    return deleted_count


def sweep_bench_rows(supabase: Any) -> int:
    """Delete leftovers of ANY bench run (e.g. a process killed before its cleanup ran)."""
    res = supabase.table("claims").delete().like("raw_input", f"{TAG_PREFIX}%").execute()
    return len(res.data or [])


def _db_host() -> Optional[str]:
    return urlparse(settings.supabase_url or "").hostname


def _check_database(confirm_db: Optional[str]) -> None:
    host = _db_host()
    if not host or confirm_db != host:
        raise BenchInputError(
            f"The bench writes tagged rows into {host!r}. Pass --confirm-db {host} to confirm that is the "
            f"project it may use (got {confirm_db!r}).")


class SeedResult(NamedTuple):
    claim_id: Optional[str]
    reason: Optional[str]       # why no row was seeded for this claim (None when claim_id is set)


async def seed_via_replay(
    texts: List[str],
    bench_run_id: str,
    cache_dir: Path | str,
    *,
    verify: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
    replay: Any = None,
) -> Dict[str, SeedResult]:
    """
    Seed one cache row per claim by running the real pipeline on recorded evidence (replay mode, so no
    quota). The persisted row's raw_input gets the bench tag so cleanup can find it; nothing else about
    the row is touched, and the claim text (hence every replay key) is unchanged.

    Raises BenchInputError on ANY replay miss: a miss means the pipeline ran without recorded evidence,
    and the row would not be what the scoring engine produces from real evidence.
    """
    from app.agents import orchestrator
    from app.agents.orchestrator import AllAgentsFailedError
    from eval.replay import EvidenceReplay

    verify = verify or orchestrator.verify_claim
    replay = replay or EvidenceReplay(Path(cache_dir), "replay")
    original_insert = orchestrator.insert_cache

    async def tagged_insert(**kwargs: Any) -> Optional[str]:
        kwargs["raw_input"] = f"{_tag(bench_run_id)} {kwargs['raw_input']}"
        return await original_insert(**kwargs)

    results: Dict[str, SeedResult] = {}
    replay.install()
    orchestrator.insert_cache = tagged_insert
    try:
        for text in texts:
            replay.reset_claim_counts(text)
            try:
                response = await verify(text, bypass_cache=False)
            except AllAgentsFailedError:
                results[text] = SeedResult(None, "every agent failed")
                continue
            if replay.claim_misses():
                raise BenchInputError(
                    f"Replay miss while seeding {text!r}: that claim is not in the evidence cache "
                    f"({replay.claim_misses()} missing call(s)). Pairs must be claims the record run recorded.")
            if response.get("cached"):
                results[text] = SeedResult(None, "already served from the cache (not a row the bench seeded)")
            elif not response.get("id"):
                results[text] = SeedResult(None, "the pipeline did not persist a row")
            else:
                results[text] = SeedResult(response["id"], None)
    finally:
        orchestrator.insert_cache = original_insert
        replay.uninstall()
    return results


async def _timed_lookup(
    text: str, embed: Callable[[str], Awaitable[List[float]]], lookup: Callable[..., Awaitable[Any]],
    seeded_ids: set,
    expected_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One request-path cache probe: embed + lookup. A hit counts only when it is the row seeded for THIS
    claim (expected_id); another seeded row is a wrong-row hit, anything else a foreign row. For the
    negative variants expected_id is None and any seeded row is a (false-positive) hit.
    """
    t0 = time.perf_counter()
    embedding = await embed(text)
    t1 = time.perf_counter()
    hit = await lookup(embedding, text)
    t2 = time.perf_counter()
    got = hit.get("claim_id") if hit else None
    in_seed = got in seeded_ids
    return {
        "hit": bool(hit) and in_seed and (expected_id is None or got == expected_id),
        "wrong_row_hit": bool(hit) and in_seed and expected_id is not None and got != expected_id,
        "foreign_hit": bool(hit) and not in_seed,
        "total_ms": (t2 - t0) * 1000,
        "lookup_ms": (t2 - t1) * 1000,
    }


async def run_cache_benchmark(
    record_summary_path: Path | str = DEFAULT_RECORD_SUMMARY,
    output_path: Path | str = DEFAULT_OUT,
    pairs: Optional[List[Dict[str, str]]] = None,
    negatives: Optional[List[Dict[str, str]]] = None,
    *,
    evidence_cache: Path | str = DEFAULT_EVIDENCE_CACHE,
    confirm_db: Optional[str] = None,
    supabase: Any = None,
    embed: Optional[Callable[[str], List[float]]] = None,
    verify: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
    replay: Any = None,
    lookup: Optional[Callable[..., Awaitable[Any]]] = None,
) -> Dict[str, Any]:
    """
    1. Reads the live p50 and calls per claim from a valid record run (refuses otherwise).
    2. Seeds one row per pair through the real pipeline on recorded evidence (replay, no quota); a replay
       miss aborts. Rows are tagged so cleanup can find them.
    3. Probes each exact claim. A pair counts only if the cache serves ITS row for the exact claim; fewer
       than MIN_SERVED_PAIRS such pairs aborts (nothing is reported from a bench that did not measure).
    4. Times embed + lookup for each served pair's exact claim and paraphrase; probes the negatives of
       served pairs, which must all MISS.
    5. Cleans up every tagged row in `finally`; a cleanup failure never masks the original error.
    """
    pairs = pairs or []
    negatives = negatives or []
    baseline = load_live_baseline(record_summary_path)
    _check_database(confirm_db)
    if len(pairs) < MIN_SERVED_PAIRS:
        raise BenchInputError(f"Need >= {MIN_SERVED_PAIRS} pairs (use load_pairs / --pairs), got {len(pairs)}.")

    if supabase is None:
        from app.db.connection import get_supabase
        supabase = get_supabase()
    if lookup is None:
        from app.services.cache import lookup_cache
        lookup = lookup_cache
    if embed is None:
        from app.services.embedding import generate_embedding
        embed = generate_embedding

    async def embed_async(text: str) -> List[float]:
        return await asyncio.to_thread(embed, text)

    bench_run_id = uuid.uuid4().hex[:8]
    created_claim_ids: List[str] = []
    print(f"\n--- Starting Cache Benchmark (run id: {bench_run_id}, pairs={len(pairs)}, db: {_db_host()}) ---")

    try:
        seeds = await seed_via_replay([p["claim"] for p in pairs], bench_run_id, evidence_cache,
                                      verify=verify, replay=replay)
        created_claim_ids.extend(r.claim_id for r in seeds.values() if r.claim_id)
        seeded_ids = set(created_claim_ids)
        own = {p["id"]: seeds[p["claim"]].claim_id for p in pairs}
        print(f"Seeded {len(seeded_ids)} of {len(pairs)} claims through the pipeline on recorded evidence.")

        # A row the cache will not serve (low confidence, a degraded agent, ...) is not benchmarkable.
        exact: Dict[str, Dict[str, Any]] = {}
        for p in pairs:
            if own[p["id"]]:
                exact[p["id"]] = await _timed_lookup(p["claim"], embed_async, lookup, seeded_ids, own[p["id"]])
        served = [p for p in pairs if exact.get(p["id"], {}).get("hit")]
        not_served = [
            {"id": p["id"],
             "reason": seeds[p["claim"]].reason or "seeded, but the cache did not serve it for its exact claim"}
            for p in pairs if p not in served
        ]
        if len(served) < MIN_SERVED_PAIRS:
            raise BenchInputError(
                f"Only {len(served)} of {len(pairs)} seeded rows are served by the cache (need >= "
                f"{MIN_SERVED_PAIRS}). Not served: {not_served[:5]}{' ...' if len(not_served) > 5 else ''}. "
                "Choose more recorded claims with complete, non-low-confidence results.")

        para = [await _timed_lookup(p["paraphrase"], embed_async, lookup, seeded_ids, own[p["id"]]) for p in served]
        served_ids = {p["id"] for p in served}
        probed_negatives = [n for n in negatives if n["of"] in served_ids]
        neg = [await _timed_lookup(n["text"], embed_async, lookup, seeded_ids) for n in probed_negatives]
        exact_results = [exact[p["id"]] for p in served]

        hits = [r for r in exact_results + para if r["hit"]]
        summary = summarize([r["total_ms"] for r in hits], baseline["live_p50_ms"], baseline["calls_per_claim"])
        false_positives = sum(1 for r in neg if r["hit"])

        def block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            n_hits = sum(1 for r in results if r["hit"])
            return {
                "n": len(results),
                "hits": n_hits,
                "hit_rate": n_hits / len(results) if results else None,
                "wrong_row_hits": sum(1 for r in results if r["wrong_row_hit"]),
                "foreign_hits_ignored": sum(1 for r in results if r["foreign_hit"]),
                "latencies_ms": [round(r["total_ms"], 2) for r in results],
                "lookup_only_ms": [round(r["lookup_ms"], 2) for r in results],
            }

        full_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bench_run_id": bench_run_id,
            "database": _db_host(),
            "pairs_requested": len(pairs),
            "seeded_rows": len(seeded_ids),
            "pairs_served": len(served),
            "pairs_not_served": not_served,
            "negatives_probed": len(probed_negatives),
            "negatives_skipped_unserved_pair": len(negatives) - len(probed_negatives),
            "seeding": "verify_claim(bypass_cache=False) under EvidenceReplay replay on recorded evidence",
            "latency_includes": "embedding + pgvector lookup + equivalence check",
            "baseline": baseline,
            "exact_lookups": block(exact_results),
            "paraphrase_lookups": block(para),
            "negative_lookups": {**block(neg), "false_positives": false_positives,
                                 "passed": false_positives == 0},
            "summary": summary,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

        print(f"\nCache benchmark complete! Results saved to {out_path}")
        print(f"  - Pairs served by the cache: {len(served)}/{len(pairs)}")
        print(f"  - Exact hit rate:       {full_report['exact_lookups']['hits']}/{len(served)}")
        print(f"  - Paraphrase hit rate:  {full_report['paraphrase_lookups']['hits']}/{len(served)}")
        print(f"  - Negation/number false positives: {false_positives}/{len(neg)} (must be 0)")
        print(f"  - Hit p50/p95:          {summary['hit_p50_ms']} / {summary['hit_p95_ms']} ms "
              f"(live p50 {baseline['live_p50_ms']:.0f} ms from {baseline['source']}, n={baseline['n']})")
        print(f"  - Target (hit p95 < 10% of live p50): {summary['target_met']}")
        return full_report
    finally:
        print(f"Cleaning up bench rows (run id: {bench_run_id})...")
        try:
            deleted = cleanup_bench_rows(supabase, bench_run_id, created_claim_ids)
            print(f"Cleaned up {deleted} rows from Supabase.")
        except Exception:  # noqa: BLE001 - never mask the error that got us here
            logger.exception("bench cleanup FAILED; run `python eval/cache_bench.py --sweep --confirm-db %s`",
                             _db_host())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="VeriFact cache benchmark")
    parser.add_argument("--record-summary", type=Path, default=DEFAULT_RECORD_SUMMARY,
                        help="path to record_run.json (a valid, live-latency record run)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="path to write cache_bench.json")
    parser.add_argument("--confirm-db", default=None,
                        help="host of the Supabase project the bench may write tagged rows into")
    parser.add_argument("--pairs", type=Path, default=None,
                        help="JSON with pairs (recorded claims + paraphrases) and negatives; see load_pairs")
    parser.add_argument("--evidence-cache", type=Path, default=DEFAULT_EVIDENCE_CACHE,
                        help="evidence cache written by the record run, replayed to seed the rows")
    parser.add_argument("--sweep", action="store_true", help="delete leftover bench rows and exit")
    args = parser.parse_args(argv)

    try:
        _check_database(args.confirm_db)
        if args.sweep:
            from app.db.connection import get_supabase
            print(f"Swept {sweep_bench_rows(get_supabase())} leftover bench rows.")
            return
        if args.pairs is None:
            raise BenchInputError("--pairs is required: the bench seeds rows from claims the record run recorded.")
        pairs, negatives = load_pairs(args.pairs)
        asyncio.run(run_cache_benchmark(record_summary_path=args.record_summary, output_path=args.output,
                                        pairs=pairs, negatives=negatives, evidence_cache=args.evidence_cache,
                                        confirm_db=args.confirm_db))
    except BenchInputError as exc:
        print(f"Refusing to run: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
