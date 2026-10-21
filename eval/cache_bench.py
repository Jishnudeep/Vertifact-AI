"""
Cache benchmark with counted call savings (Plan 9, Task 6 / Gate D4).

Measures semantic cache hit latency (embedding + pgvector lookup + NLI equivalence, the same path
verify_claim takes) for exact claims and faithful paraphrases, and checks that negated or
number-changed variants MISS. The comparison baseline (live p50, calls per claim) comes only from a
valid record run; there are no fallback constants, and a figure that was not measured is None.

It writes tagged seed rows into the configured Supabase project, so it refuses to run unless the
operator names that project (--confirm-db <host>), and it always cleans up in `finally`.
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
from typing import Any, Awaitable, Callable, Dict, List, Optional
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

TAG_PREFIX = "[bench:"
# Latency from these modes is a live measurement; a replay or record-missing run mixes cached turns in.
LIVE_LATENCY_MODES = {"record", "live"}


class BenchInputError(RuntimeError):
    """The bench cannot produce an honest figure (or may not touch this database); nothing was run."""


# 20 pairs of claims and faithful paraphrases for the benchmark
BENCH_PAIRS: List[Dict[str, str]] = [
    {
        "id": "bench-01",
        "claim": "NASA confirmed asteroid Apophis will safely pass Earth in 2029 without colliding.",
        "paraphrase": "NASA confirmed that asteroid Apophis will fly past Earth safely in 2029 with no chance of impact.",
    },
    {
        "id": "bench-02",
        "claim": "The Federal Reserve announced it is not replacing physical dollar bills with a CBDC.",
        "paraphrase": "The Federal Reserve stated that physical dollar bills will not be replaced by a central bank digital currency.",
    },
    {
        "id": "bench-03",
        "claim": "The WHO declared the end of the global COVID-19 public health emergency in May 2023.",
        "paraphrase": "In May 2023, the World Health Organization officially declared the end to the COVID-19 global emergency.",
    },
    {
        "id": "bench-04",
        "claim": "The European Union approved rules mandating zero-emission sales for new cars starting in 2035.",
        "paraphrase": "EU legislation requires all new vehicles sold starting in 2035 to produce zero emissions.",
    },
    {
        "id": "bench-05",
        "claim": "Finland officially joined NATO as its 31st member nation on April 4, 2023.",
        "paraphrase": "On April 4, 2023, Finland was formally admitted into NATO as the alliance's 31st member.",
    },
    {
        "id": "bench-06",
        "claim": "The Great Wall of China cannot be seen from low Earth orbit with the unaided eye.",
        "paraphrase": "Astronauts confirm the Great Wall of China is not visible from low Earth orbit without binoculars or a telescope.",
    },
    {
        "id": "bench-07",
        "claim": "Drinking raw unpasteurized milk does not cure lactose intolerance.",
        "paraphrase": "Medical experts report that consuming unpasteurized raw milk does not cure or relieve lactose intolerance.",
    },
    {
        "id": "bench-08",
        "claim": "A 2022 study detected microplastic polymer particles in human blood samples.",
        "paraphrase": "Microplastics were found in human bloodstream samples in a 2022 research investigation.",
    },
    {
        "id": "bench-09",
        "claim": "Eating carrots does not improve normal night vision beyond baseline levels.",
        "paraphrase": "Consuming carrots will not grant superior night vision to people with normal vitamin A levels.",
    },
    {
        "id": "bench-10",
        "claim": "Thermal expansion increases the Eiffel Tower's height by up to 15 centimeters in hot summer weather.",
        "paraphrase": "Summer heat causes the Eiffel Tower to expand thermally and gain up to 15 centimeters in height.",
    },
    {
        "id": "bench-11",
        "claim": "Human industrial emissions exceed annual volcanic carbon dioxide emissions by at least eighty times.",
        "paraphrase": "Annual human CO2 emissions are at least 80 times larger than all volcanic emissions combined.",
    },
    {
        "id": "bench-12",
        "claim": "Bananas naturally contain potassium-40 making them slightly radioactive but harmless.",
        "paraphrase": "Because of natural potassium-40, bananas emit trace amounts of radiation that pose no health danger.",
    },
    {
        "id": "bench-13",
        "claim": "Lightning strikes the Empire State Building dozens of times every year.",
        "paraphrase": "The Empire State Building is hit by lightning multiple dozens of times each year.",
    },
    {
        "id": "bench-14",
        "claim": "The Panama Canal officially opened for maritime navigation in August 1914.",
        "paraphrase": "In August 1914, the Panama Canal began its official commercial maritime operations.",
    },
    {
        "id": "bench-15",
        "claim": "James Webb Space Telescope detected carbon-bearing molecules on exoplanet K2-18b in 2023.",
        "paraphrase": "In 2023, observations from the James Webb Space Telescope identified carbon-rich molecules in exoplanet K2-18b's atmosphere.",
    },
    {
        "id": "bench-16",
        "claim": "Water boils at 100 degrees Celsius at standard atmospheric sea level pressure.",
        "paraphrase": "At 1 atmosphere of sea-level pressure, the boiling point of pure water is 100 degrees Celsius.",
    },
    {
        "id": "bench-17",
        "claim": "The ozone layer above Antarctica has shown continuous signs of recovery under the Montreal Protocol.",
        "paraphrase": "Thanks to the Montreal Protocol, the Antarctic ozone hole continues on a path toward gradual healing.",
    },
    {
        "id": "bench-18",
        "claim": "The speed of light in a vacuum is approximately 300,000 kilometers per second.",
        "paraphrase": "In a complete vacuum, light travels at a velocity of roughly 300,000 kilometers per second.",
    },
    {
        "id": "bench-19",
        "claim": "Honey found in ancient Egyptian tombs remained unspoiled and edible after thousands of years.",
        "paraphrase": "Centuries-old honey recovered from Egyptian pyramids and tombs remained preserved without spoiling.",
    },
    {
        "id": "bench-20",
        "claim": "Antibiotics kill bacteria and have no effect against viral infections such as influenza.",
        "paraphrase": "Antibiotic medications target bacterial organisms only and are ineffective against viral diseases like the flu.",
    },
]


# Variants that must NOT be served from the cache: negation, changed number, changed entity.
# Embeddings alone cannot tell these from the original, so the equivalence check has to.
BENCH_NEGATIVES: List[Dict[str, str]] = [
    {"id": "neg-01", "of": "bench-01",
     "text": "NASA confirmed asteroid Apophis will collide with Earth in 2029."},
    {"id": "neg-02", "of": "bench-03",
     "text": "The WHO declared the end of the global COVID-19 public health emergency in May 2013."},
    {"id": "neg-03", "of": "bench-05",
     "text": "Finland officially joined NATO as its 41st member nation on April 4, 2023."},
    {"id": "neg-04", "of": "bench-07",
     "text": "Drinking raw unpasteurized milk cures lactose intolerance."},
    {"id": "neg-05", "of": "bench-14",
     "text": "The Panama Canal officially opened for maritime navigation in August 1941."},
    {"id": "neg-06", "of": "bench-20",
     "text": "Antibiotics kill bacteria and also cure viral infections such as influenza."},
]


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


def _seed_row(bench_run_id: str, text: str, embedding: List[float]) -> Dict[str, Any]:
    """A complete, servable cached result that says plainly it is a benchmark seed and carries no sources."""
    return dict(
        input_type="text",
        raw_input=f"{_tag(bench_run_id)} {text}",
        extracted_claim=text,
        embedding=embedding,
        c_total=0.5,
        s_fact=0.5,
        b_ling=0.5,
        v_consensus=0.0,
        weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
        confidence_level="medium",
        agent_outputs={name: {"degraded": False, "note": "bench_seed"}
                       for name in ("factcheck", "crossref", "linguistic")},
        source_citations=[],
        linguistic_profile={},
        processing_time_ms=0,
        warnings=["Benchmark seed row, not a real analysis."],
        v_refute=0.0,
        evidence_status="unverified",
    )


async def _timed_lookup(
    text: str, embed: Callable[[str], Awaitable[List[float]]], lookup: Callable[..., Awaitable[Any]],
    seeded_ids: set,
) -> Dict[str, Any]:
    """One request-path cache probe: embed + lookup. A hit only counts when it is one of OUR seeded rows."""
    t0 = time.perf_counter()
    embedding = await embed(text)
    t1 = time.perf_counter()
    hit = await lookup(embedding, text)
    t2 = time.perf_counter()
    return {
        "hit": bool(hit) and hit.get("claim_id") in seeded_ids,
        "foreign_hit": bool(hit) and hit.get("claim_id") not in seeded_ids,
        "total_ms": (t2 - t0) * 1000,
        "lookup_ms": (t2 - t1) * 1000,
    }


async def run_cache_benchmark(
    record_summary_path: Path | str = DEFAULT_RECORD_SUMMARY,
    output_path: Path | str = DEFAULT_OUT,
    pairs: Optional[List[Dict[str, str]]] = None,
    *,
    confirm_db: Optional[str] = None,
    supabase: Any = None,
    embed: Optional[Callable[[str], List[float]]] = None,
    insert: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
    lookup: Optional[Callable[..., Awaitable[Any]]] = None,
) -> Dict[str, Any]:
    """
    1. Reads the live p50 and calls per claim from a valid record run (refuses otherwise).
    2. Seeds N servable, tagged rows (production embedding path).
    3. Times embed + lookup for each exact claim and each paraphrase; probes the negative variants.
    4. Cleans up every tagged row in `finally`; a cleanup failure never masks the original error.
    """
    pairs = pairs or BENCH_PAIRS
    baseline = load_live_baseline(record_summary_path)
    _check_database(confirm_db)

    if supabase is None:
        from app.db.connection import get_supabase
        supabase = get_supabase()
    if insert is None or lookup is None:
        from app.services.cache import insert_cache, lookup_cache
        insert, lookup = insert or insert_cache, lookup or lookup_cache
    if embed is None:
        from app.services.embedding import generate_embedding
        embed = generate_embedding

    async def embed_async(text: str) -> List[float]:
        return await asyncio.to_thread(embed, text)

    bench_run_id = uuid.uuid4().hex[:8]
    created_claim_ids: List[str] = []
    print(f"\n--- Starting Cache Benchmark (run id: {bench_run_id}, n={len(pairs)}, db: {_db_host()}) ---")

    try:
        for item in pairs:
            row = _seed_row(bench_run_id, item["claim"], await embed_async(item["claim"]))
            cid = await insert(**row)
            if cid:
                created_claim_ids.append(cid)
        seeded_ids = set(created_claim_ids)
        print(f"Seeded {len(seeded_ids)} claims.")

        exact = [await _timed_lookup(p["claim"], embed_async, lookup, seeded_ids) for p in pairs]
        para = [await _timed_lookup(p["paraphrase"], embed_async, lookup, seeded_ids) for p in pairs]
        negatives = [await _timed_lookup(n["text"], embed_async, lookup, seeded_ids) for n in BENCH_NEGATIVES]

        hits = [r for r in exact + para if r["hit"]]
        summary = summarize([r["total_ms"] for r in hits], baseline["live_p50_ms"], baseline["calls_per_claim"])
        false_positives = sum(1 for r in negatives if r["hit"])

        def block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            n_hits = sum(1 for r in results if r["hit"])
            return {
                "n": len(results),
                "hits": n_hits,
                "hit_rate": n_hits / len(results) if results else None,
                "foreign_hits_ignored": sum(1 for r in results if r["foreign_hit"]),
                "latencies_ms": [round(r["total_ms"], 2) for r in results],
                "lookup_only_ms": [round(r["lookup_ms"], 2) for r in results],
            }

        full_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bench_run_id": bench_run_id,
            "database": _db_host(),
            "pairs_tested": len(pairs),
            "latency_includes": "embedding + pgvector lookup + equivalence check",
            "baseline": baseline,
            "exact_lookups": block(exact),
            "paraphrase_lookups": block(para),
            "negative_lookups": {**block(negatives), "false_positives": false_positives,
                                 "passed": false_positives == 0},
            "summary": summary,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

        print(f"\nCache benchmark complete! Results saved to {out_path}")
        print(f"  - Exact hit rate:       {full_report['exact_lookups']['hits']}/{len(pairs)}")
        print(f"  - Paraphrase hit rate:  {full_report['paraphrase_lookups']['hits']}/{len(pairs)}")
        print(f"  - Negation/number false positives: {false_positives}/{len(negatives)} (must be 0)")
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
    parser.add_argument("--sweep", action="store_true", help="delete leftover bench rows and exit")
    args = parser.parse_args(argv)

    try:
        _check_database(args.confirm_db)
        if args.sweep:
            from app.db.connection import get_supabase
            print(f"Swept {sweep_bench_rows(get_supabase())} leftover bench rows.")
            return
        asyncio.run(run_cache_benchmark(record_summary_path=args.record_summary, output_path=args.output,
                                        confirm_db=args.confirm_db))
    except BenchInputError as exc:
        print(f"Refusing to run: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
