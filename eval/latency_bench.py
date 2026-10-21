"""
Live latency benchmark for the ADK-native worker agents (benchmark-then-optimize, phase 1).

Runs `verify_claim(bypass_cache=True)` LIVE over a handful of curated claims, one at a time with a
configurable pause, and records where the time goes:

  per claim, per agent : wall time (the pipeline's own `agents[].duration_ms`)
  per LLM turn         : latency, prompt/completion tokens (when the provider reports them)
  per tool call        : latency, by tool name
  429 retries          : counted from google-genai retry/rate-limit log lines, attributed to the
                         LLM turn whose time window they fall in (approximate: agents run in
                         parallel, so two turns can share a window)

and writes `eval/results/latency_bench.json` (covered by the `eval/results/*.json` ignore rule)
plus a printed p50/p95 table that names the critical path: the agent that most often finishes last
and therefore sets the request latency.

It spends real quota (Gemini LLM turns, Gemini grounded searches, Fact Check API) and is NOT run by
any test. Run from the repo root:

    uv run --project backend python eval/latency_bench.py --n 15 --delay 6

The daily spend budgets are disabled for the run (as in run_eval.py) so the numbers measure the
pipeline, not the budget guard. Nothing here changes production code: it wraps the model seam
(`app.agents.common.get_model`) and the agents' callbacks (`common.agent_kwargs`) from the outside.
"""
import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
for entry in (BASE_DIR, BASE_DIR / "backend"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

# Must be set before app.config is imported; an explicit environment value still wins.
os.environ.setdefault("SEARCH_DAILY_CALL_BUDGET", "0")
os.environ.setdefault("LLM_DAILY_CALL_BUDGET", "0")

from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

import app.agents.common as common  # noqa: E402
from app.agents.orchestrator import verify_claim  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.bias_lookup import load_bias_ratings  # noqa: E402
from app.services.warmup import ensure_models_warm  # noqa: E402

DEFAULT_DATASET = BASE_DIR / "eval" / "datasets" / "curated_claims_seed.json"
DEFAULT_OUTPUT = BASE_DIR / "eval" / "results" / "latency_bench.json"
AGENTS = ("factcheck", "crossref", "linguistic")

logger = logging.getLogger("latency_bench")

# Which agent a model turn / tool call belongs to.
_INSTRUCTION_MARKERS = {
    "fact-check lookup agent": "factcheck",
    "cross-reference agent": "crossref",
    "linguistic analysis assistant": "linguistic",
}
_TOOL_AGENT = {
    "search_fact_check_database": "factcheck",
    "search_fact_check_sites": "factcheck",
    "search_news_coverage": "crossref",
}


# --- instrumentation -----------------------------------------------------------------------------

class Recorder:
    """Events of the claim currently running. The benchmark is sequential, so one global is enough."""

    def __init__(self) -> None:
        self.llm_turns: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.rate_limit_log_lines = 0

    def reset(self) -> None:
        self.llm_turns.clear()
        self.tool_calls.clear()
        self.rate_limit_log_lines = 0


RECORDER = Recorder()


class _RateLimitLogCounter(logging.Handler):
    """Counts google-genai/ADK log lines that mention a rate limit or a retry (the 429 retry signal)."""

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage().lower()
        if "429" in text or "ratelimit" in text or "rate limit" in text or "retrying" in text:
            RECORDER.rate_limit_log_lines += 1


class TimedLlm(BaseLlm):
    """Wraps the worker model and times every turn."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "timed"
    inner: object

    @property
    def capabilities(self):
        return self.inner.capabilities

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        instruction = str(getattr(llm_request.config, "system_instruction", "") or "")
        agent = next((a for marker, a in _INSTRUCTION_MARKERS.items() if marker in instruction), "unknown")
        retries_before = RECORDER.rate_limit_log_lines
        started = time.perf_counter()
        usage = None
        try:
            async for response in self.inner.generate_content_async(llm_request, stream=stream):
                if getattr(response, "usage_metadata", None) is not None:
                    usage = response.usage_metadata
                yield response
        finally:
            RECORDER.llm_turns.append({
                "agent": agent,
                "ms": round((time.perf_counter() - started) * 1000),
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "rate_limit_retries": RECORDER.rate_limit_log_lines - retries_before,
            })


_TOOL_START: Dict[str, float] = {}


def _before_tool(tool, args, tool_context):
    _TOOL_START[tool_context.function_call_id] = time.perf_counter()


def _after_tool(tool, args, tool_context, tool_response):
    started = _TOOL_START.pop(tool_context.function_call_id, None)
    if started is not None:
        RECORDER.tool_calls.append({
            "agent": _TOOL_AGENT.get(tool.name, "unknown"),
            "tool": tool.name,
            "ms": round((time.perf_counter() - started) * 1000),
        })


def install_instrumentation() -> None:
    original_model, original_kwargs = common.get_model, common.agent_kwargs

    def timed_model() -> BaseLlm:
        inner = original_model()
        inner = Gemini(model=inner) if isinstance(inner, str) else inner
        return TimedLlm(model=inner.model, inner=inner)

    def timed_kwargs() -> Dict[str, Any]:
        return {**original_kwargs(), "before_tool_callback": _before_tool, "after_tool_callback": _after_tool}

    common.get_model, common.agent_kwargs = timed_model, timed_kwargs
    handler = _RateLimitLogCounter()
    for name in ("google_genai", "google.genai", "google_adk", "tenacity"):
        logging.getLogger(name).addHandler(handler)


# --- statistics ----------------------------------------------------------------------------------

def percentile(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def p50_p95(values: List[float]) -> Dict[str, Optional[float]]:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}


def select_claims(dataset_path: Path, n: int) -> List[Dict[str, Any]]:
    """n verifiable claims, evenly spread over the dataset (deterministic)."""
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    verifiable = [c for c in items if c.get("claim_type", "verifiable") == "verifiable"]
    if n >= len(verifiable):
        return verifiable
    step = len(verifiable) / n
    return [verifiable[int(i * step)] for i in range(n)]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    summary: Dict[str, Any] = {
        "claims": len(records),
        "failed": len(records) - len(ok),
        "request_ms": p50_p95([r["request_ms"] for r in ok]),
        "agent_ms": {a: p50_p95([r["agents"][a]["wall_ms"] for r in ok if r["agents"].get(a, {}).get("wall_ms") is not None])
                     for a in AGENTS},
        "llm_turn_ms": {a: p50_p95([t["ms"] for r in ok for t in r["agents"].get(a, {}).get("llm_turns", [])])
                        for a in AGENTS},
        "llm_turns_per_claim": {a: p50_p95([len(r["agents"].get(a, {}).get("llm_turns", [])) for r in ok])
                                for a in AGENTS},
        "tool_ms": {},
        "tokens": {"prompt": sum(t.get("prompt_tokens") or 0 for r in ok for a in AGENTS
                                 for t in r["agents"].get(a, {}).get("llm_turns", [])),
                   "completion": sum(t.get("completion_tokens") or 0 for r in ok for a in AGENTS
                                     for t in r["agents"].get(a, {}).get("llm_turns", []))},
        "rate_limit_retries": sum(t.get("rate_limit_retries", 0) for r in ok for a in AGENTS
                                  for t in r["agents"].get(a, {}).get("llm_turns", [])),
    }
    by_tool: Dict[str, List[float]] = defaultdict(list)
    for r in ok:
        for a in AGENTS:
            for call in r["agents"].get(a, {}).get("tool_calls", []):
                by_tool[call["tool"]].append(call["ms"])
    summary["tool_ms"] = {tool: p50_p95(v) for tool, v in sorted(by_tool.items())}

    # Critical path: the agents run in parallel, so the request waits for the slowest one.
    slowest: Dict[str, int] = defaultdict(int)
    for r in ok:
        walls = {a: r["agents"][a]["wall_ms"] for a in AGENTS if r["agents"].get(a, {}).get("wall_ms") is not None}
        if walls:
            slowest[max(walls, key=walls.get)] += 1
    top = max(slowest, key=slowest.get) if slowest else None
    summary["critical_path"] = {
        "slowest_agent_counts": dict(slowest),
        "agent": top,
        "median_agent_ms": summary["agent_ms"][top]["p50"] if top else None,
        "median_request_ms": summary["request_ms"]["p50"],
    }
    return summary


def print_table(summary: Dict[str, Any]) -> None:
    def fmt(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:,.0f}"

    print("\n" + "=" * 78)
    print(f" Latency benchmark: {summary['claims']} claims ({summary['failed']} failed)")
    print("=" * 78)
    print(f" {'':<22}{'n':>5}{'p50 ms':>12}{'p95 ms':>12}")
    r = summary["request_ms"]
    print(f" {'request (end to end)':<22}{r['n']:>5}{fmt(r['p50']):>12}{fmt(r['p95']):>12}")
    for a in AGENTS:
        s = summary["agent_ms"][a]
        print(f" {'agent ' + a:<22}{s['n']:>5}{fmt(s['p50']):>12}{fmt(s['p95']):>12}")
    for a in AGENTS:
        s = summary["llm_turn_ms"][a]
        if s["n"]:
            print(f" {'llm turn ' + a:<22}{s['n']:>5}{fmt(s['p50']):>12}{fmt(s['p95']):>12}")
    for tool, s in summary["tool_ms"].items():
        print(f" {'tool ' + tool[:16]:<22}{s['n']:>5}{fmt(s['p50']):>12}{fmt(s['p95']):>12}")
    print(f"\n tokens: {summary['tokens']}   rate-limit retries seen: {summary['rate_limit_retries']}")
    cp = summary["critical_path"]
    if cp["agent"]:
        print(f" CRITICAL PATH: '{cp['agent']}' (median {fmt(cp['median_agent_ms'])} ms of a "
              f"{fmt(cp['median_request_ms'])} ms request); slowest-agent counts {cp['slowest_agent_counts']}")
    print("=" * 78)


# --- the run -------------------------------------------------------------------------------------

async def run_bench(dataset: Path, n: int, delay_s: float, output: Path) -> Dict[str, Any]:
    load_bias_ratings()
    await asyncio.to_thread(ensure_models_warm)         # a cold model load is not pipeline latency
    install_instrumentation()
    claims = select_claims(dataset, n)
    records: List[Dict[str, Any]] = []

    for index, item in enumerate(claims):
        if index and delay_s:
            await asyncio.sleep(delay_s)                # stay under the provider's per-minute limits
        RECORDER.reset()
        record: Dict[str, Any] = {"id": item["id"], "claim": item["claim_text"]}
        started = time.perf_counter()
        try:
            response = await verify_claim(item["claim_text"], bypass_cache=True)
            record["request_ms"] = round((time.perf_counter() - started) * 1000)
            by_name = {a["name"]: a for a in response.get("agents", [])}
            record["agents"] = {}
            for agent in AGENTS:
                status = by_name.get(agent, {})
                record["agents"][agent] = {
                    "wall_ms": status.get("duration_ms"),
                    "status": status.get("status"),
                    "error_code": status.get("error_code"),
                    "llm_turns": [t for t in RECORDER.llm_turns if t["agent"] == agent],
                    "tool_calls": [t for t in RECORDER.tool_calls if t["agent"] == agent],
                }
            record["c_total"] = response["trust_score"]["c_total"]
        except Exception as exc:  # noqa: BLE001 - a failed claim is data, not a crash
            record["request_ms"] = round((time.perf_counter() - started) * 1000)
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        logger.info("[%d/%d] %s -> %s ms%s", index + 1, len(claims), item["id"], record["request_ms"],
                    f" ERROR {record['error']}" if record.get("error") else "")

    summary = summarize(records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model_id,
        "search_model": settings.search_model_id,
        "agent_timeout_s": settings.agent_timeout_s,
        "request_deadline_s": settings.request_deadline_s,
        "delay_s": delay_s,
        "summary": summary,
        "claims": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(summary)
    print(f"\n Wrote {output}")
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live latency benchmark for the VeriFact worker agents.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="EvalClaim JSON file.")
    parser.add_argument("--n", type=int, default=15, help="Number of verifiable claims (evenly spread).")
    parser.add_argument("--delay", type=float, default=6.0,
                        help="Seconds to pause between claims (provider rate limits).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true",
                        help="Required. Confirms you accept spending live Gemini quota (about 8 requests per claim).")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error(f"refusing to run without --live: {estimated_requests(args.n)} requests of free-tier quota "
                     "(500 RPD) would be spent.")
    return args


REQUESTS_PER_CLAIM = 8      # cold claim, bench 2026-10-20


def estimated_requests(n: int) -> int:
    return REQUESTS_PER_CLAIM * n


def print_estimate(n: int) -> None:
    print(f"latency_bench will spend about {estimated_requests(n)} Gemini requests "
          f"({REQUESTS_PER_CLAIM} per claim x {n}) of the 500 requests/day free-tier cap.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()
    print_estimate(args.n)
    asyncio.run(run_bench(args.dataset, args.n, args.delay, args.output))
