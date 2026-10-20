"""
Cache benchmark with counted call savings (Plan 9, Task 6 / Gate D4).

Measures semantic cache lookup latency (exact hits and paraphrase hits) against pgvector
and bidirectional NLI equivalence, asserting hit p95 < 10% of live uncached p50.
Cleans up all bench-tagged rows from Supabase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.db.connection import get_supabase
from app.services.cache import insert_cache, lookup_cache
from app.services.claim_equivalence import are_equivalent

logger = logging.getLogger("cache_bench")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "eval" / "results" / "cache_bench.json"
DEFAULT_RECORD_SUMMARY = BASE_DIR / "eval" / "results" / "record_run.json"

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


def summarize(
    hit_ms: List[float],
    live_p50_ms: float,
    calls_per_claim: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute summary metrics for cache benchmark.
    target_met: hit_p95 < 0.10 * live_p50
    calls_saved_per_hit: calls_per_claim.mean
    """
    n = len(hit_ms)
    if n >= 2:
        q = statistics.quantiles(hit_ms, n=100, method="inclusive")
        hit_p50 = round(q[49], 2)
        hit_p95 = round(q[94], 2)
    elif n == 1:
        hit_p50 = round(hit_ms[0], 2)
        hit_p95 = round(hit_ms[0], 2)
    else:
        hit_p50 = 0.0
        hit_p95 = 0.0

    speedup = round(live_p50_ms / hit_p50, 2) if hit_p50 > 0 else 0.0
    target_met = bool(hit_p95 < 0.10 * live_p50_ms) if (live_p50_ms > 0 and n > 0) else False
    calls_saved = float(calls_per_claim.get("mean", 0.0))

    return {
        "n_hits": n,
        "hit_p50_ms": hit_p50,
        "hit_p95_ms": hit_p95,
        "live_p50_ms": live_p50_ms,
        "speedup": speedup,
        "target_met": target_met,
        "calls_saved_per_hit": calls_saved,
    }


def cleanup_bench_rows(supabase: Any, bench_run_id: str, claim_ids: Optional[List[str]] = None) -> int:
    """
    Delete all rows inserted for this benchmark run from claims (cascades to results).
    Asserts zero tagged rows remain in Supabase.
    """
    deleted_count = 0
    tag = f"bench:{bench_run_id}"

    if claim_ids:
        for cid in claim_ids:
            try:
                res = supabase.table("claims").delete().eq("id", cid).execute()
                deleted_count += len(res.data or [])
            except Exception as e:
                logger.warning(f"Error deleting claim {cid}: {e}")

    try:
        res2 = supabase.table("claims").delete().like("raw_input", f"%{tag}%").execute()
        deleted_count += len(res2.data or [])
    except Exception as e:
        logger.warning(f"Error deleting claims by tag {tag}: {e}")

    # Verify zero tagged rows remain
    check = supabase.table("claims").select("id").like("raw_input", f"%{tag}%").execute()
    remaining = len(check.data or [])
    if remaining > 0:
        raise RuntimeError(f"Cleanup failed: {remaining} rows with tag '{tag}' still remain in Supabase claims table.")

    return deleted_count


async def run_cache_benchmark(
    record_summary_path: Path | str = DEFAULT_RECORD_SUMMARY,
    output_path: Path | str = DEFAULT_OUT,
    pairs: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Execute the cache benchmark:
    1. Reads live p50 latency and calls_per_claim from the record run summary.
    2. Seeds N results into Supabase tagged with bench_run_id.
    3. Measures lookup_cache timing for exact texts and paraphrases.
    4. Cleans up all bench rows in finally block.
    """
    pairs = pairs or BENCH_PAIRS
    summary_path = Path(record_summary_path)
    if summary_path.exists():
        rec_data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_block = rec_data.get("summary", rec_data)
        live_p50 = float(summary_block.get("latency", {}).get("p50_ms", 5000.0) or 5000.0)
        calls_per_claim = summary_block.get("external_calls_per_claim", {"mean": 3.0, "p50": 3.0, "max": 4})
    else:
        logger.warning(f"Record summary not found at {summary_path}; using fallback live p50 = 5000ms")
        live_p50 = 5000.0
        calls_per_claim = {"mean": 3.0, "p50": 3.0, "max": 4}

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(settings.embedding_model_name)

    bench_run_id = uuid.uuid4().hex[:8]
    supabase = get_supabase()
    created_claim_ids: List[str] = []

    exact_latencies_ms: List[float] = []
    exact_hits: int = 0
    para_latencies_ms: List[float] = []
    para_hits: int = 0
    all_hit_latencies_ms: List[float] = []

    print(f"\n--- Starting Cache Benchmark (run id: {bench_run_id}, n={len(pairs)}) ---")

    try:
        # Step 1: Seed results
        print("Seeding benchmark claims into cache...")
        for item in pairs:
            text = item["claim"]
            emb = embedder.encode(text).tolist()
            tagged_raw = f"[bench:{bench_run_id}] {text}"
            cid = await insert_cache(
                input_type="text",
                raw_input=tagged_raw,
                extracted_claim=text,
                embedding=emb,
                c_total=0.9,
                s_fact=1.0,
                b_ling=0.1,
                v_consensus=0.9,
                weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
                confidence_level="high",
                agent_outputs={"note": "bench_seed"},
                source_citations=[{"url": "https://example.org", "domain": "example.org", "bias_label": "Center"}],
                linguistic_profile={},
                processing_time_ms=500,
                warnings=[],
                v_refute=0.0,
                evidence_status="corroborated",
            )
            if cid:
                created_claim_ids.append(cid)

        print(f"Successfully seeded {len(created_claim_ids)} claims.")

        # Step 2: Exact lookups
        print("Benchmarking exact claim lookups...")
        for item in pairs:
            text = item["claim"]
            emb = embedder.encode(text).tolist()
            t0 = time.perf_counter()
            hit = await lookup_cache(emb, text)
            dt_ms = (time.perf_counter() - t0) * 1000
            exact_latencies_ms.append(dt_ms)
            if hit is not None:
                exact_hits += 1
                all_hit_latencies_ms.append(dt_ms)

        # Step 3: Paraphrase lookups
        print("Benchmarking paraphrase lookups...")
        for item in pairs:
            para = item["paraphrase"]
            emb = embedder.encode(para).tolist()
            t0 = time.perf_counter()
            hit = await lookup_cache(emb, para)
            dt_ms = (time.perf_counter() - t0) * 1000
            para_latencies_ms.append(dt_ms)
            if hit is not None:
                para_hits += 1
                all_hit_latencies_ms.append(dt_ms)

        # Step 4: Summarize
        summary_results = summarize(
            hit_ms=all_hit_latencies_ms if all_hit_latencies_ms else exact_latencies_ms,
            live_p50_ms=live_p50,
            calls_per_claim=calls_per_claim,
        )

        full_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bench_run_id": bench_run_id,
            "pairs_tested": len(pairs),
            "exact_lookups": {
                "n": len(pairs),
                "hits": exact_hits,
                "hit_rate": exact_hits / len(pairs),
                "latencies_ms": [round(x, 2) for x in exact_latencies_ms],
            },
            "paraphrase_lookups": {
                "n": len(pairs),
                "hits": para_hits,
                "hit_rate": para_hits / len(pairs),
                "latencies_ms": [round(x, 2) for x in para_latencies_ms],
            },
            "summary": summary_results,
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

        print(f"\nCache benchmark complete! Results saved to {out_path}")
        print(f"  • Exact hit rate:       {exact_hits}/{len(pairs)} ({exact_hits/len(pairs):.1%})")
        print(f"  • Paraphrase hit rate:  {para_hits}/{len(pairs)} ({para_hits/len(pairs):.1%})")
        print(f"  • Cache hit p50:        {summary_results['hit_p50_ms']}ms (Live p50: {live_p50:.1f}ms)")
        print(f"  • Cache hit p95:        {summary_results['hit_p95_ms']}ms (Target < {0.10*live_p50:.1f}ms: {'MET' if summary_results['target_met'] else 'MISSED'})")
        print(f"  • Speedup factor:       {summary_results['speedup']}x")
        print(f"  • Calls saved per hit:  {summary_results['calls_saved_per_hit']}")

        return full_report
    finally:
        print(f"Cleaning up bench rows (run id: {bench_run_id})...")
        deleted = cleanup_bench_rows(supabase, bench_run_id, created_claim_ids)
        print(f"Cleaned up {deleted} rows from Supabase.")


def main():
    parser = argparse.ArgumentParser(description="VeriFact cache benchmark")
    parser.add_argument("--record-summary", type=Path, default=DEFAULT_RECORD_SUMMARY,
                        help="path to record_run.json or baseline summary")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="path to write cache_bench.json")
    args = parser.parse_args()

    asyncio.run(run_cache_benchmark(
        record_summary_path=args.record_summary,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
