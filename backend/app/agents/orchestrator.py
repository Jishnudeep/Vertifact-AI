import asyncio
import datetime
import importlib.util
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# VeriFact does not use A2A. ADK's build_node() (called on EVERY ctx.run_node) tries
# `from google.adk.agents.remote_a2a_agent import RemoteA2aAgent` inside try/except ImportError
# (google/adk/workflow/utils/_workflow_graph_utils.py). Without the optional `a2a` package that
# import fails, a failed import is never cached, and it is re-searched and re-executed from disk
# on every call: synchronous work on the event loop, ~7x per request. Registering the module as
# absent makes that import fail instantly (it only has to precede the first workflow run).
if importlib.util.find_spec("a2a") is None:
    sys.modules.setdefault("google.adk.agents.remote_a2a_agent", None)

from google.adk.agents.context import Context  # noqa: E402  (after the A2A guard above)
from google.adk.apps import App  # noqa: E402
from google.adk.events.event import Event  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.adk.workflow import FunctionNode, NodeTimeoutError, Workflow, node  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.config import settings  # noqa: E402
from app.utils.validators import sanitize_text, validate_input_length, is_verifiable_claim  # noqa: E402
from app.services.embedding import generate_embedding  # noqa: E402
from app.services.cache import lookup_cache, insert_cache  # noqa: E402
from app.scoring.engine import ScoringEngine, ScoringResult  # noqa: E402
from app.agents import factcheck_agent, crossref_agent, linguistic_agent  # noqa: E402
from app.services.bias_lookup import get_bias_label  # noqa: E402
from app.utils.errors import classify_provider_error, degradation_message  # noqa: E402
from app.utils.text_processing import extract_domain  # noqa: E402

logger = logging.getLogger(__name__)

class InvalidInputError(ValueError):
    """Raised when input text is empty or exceeds the character limit (HTTP 400)."""
    pass

class NonVerifiableClaimError(ValueError):
    """Raised when input text is classified as an opinion or question (HTTP 422)."""
    pass

class AllAgentsFailedError(RuntimeError):
    """Raised when all verification agents fail or time out (HTTP 500)."""
    pass

# H1: request deadline plumbing
AGENT_TIMEOUT_CAP = 8.0        # the per-agent contract (CLAUDE.md §2)
MIN_AGENT_TIMEOUT = 1.0        # below this an agent cannot do useful work
PERSIST_RESERVE_S = 1.0        # time kept back for the DB write
CACHE_LOOKUP_TIMEOUT_S = 1.5   # pgvector round-trip + up to 3 NLI equivalence checks
PERSIST_MIN_WAIT_S = 0.25

# Persistence tasks that outlived their request; referenced so they are not GC'd mid-write.
_background_tasks: set = set()


def agent_budget(remaining_s: float) -> float:
    """Per-agent timeout that fits the remaining request budget."""
    return max(MIN_AGENT_TIMEOUT, min(AGENT_TIMEOUT_CAP, remaining_s - PERSIST_RESERVE_S))


async def drain_background_tasks(timeout: float) -> None:
    """Give in-flight persistence writes up to `timeout` seconds to finish (app shutdown)."""
    pending = list(_background_tasks)
    if not pending:
        return
    _, still_running = await asyncio.wait(pending, timeout=timeout)
    if still_running:
        logger.warning("%d persistence write(s) still running at shutdown; they will be lost.", len(still_running))


# ---------------------------------------------------------------------------
# Verification workflow (ADK 2.x graph).
#
#   START -> cache_lookup ─┬─ "hit"  -> format_cached
#                          └─ "miss" -> dispatch_agents -> score
#                                          ├─ ctx.run_node(factcheck)  ┐
#                                          ├─ ctx.run_node(crossref)   ├ concurrent, dynamic
#                                          └─ ctx.run_node(linguistic) ┘
#
# Contract notes:
# - The agents run as DYNAMIC child nodes of a supervisor (adk.dev/graphs/dynamic, "partial-failure
#   handling"). A failed or timed-out child surfaces in the supervisor as DynamicNodeFailError,
#   which is mapped to that agent's degraded neutral result; the healthy siblings finish and the
#   workflow continues (CLAUDE.md §4.3). A static fan-out (edges + JoinNode) or a ParallelWorker
#   would instead fail the whole workflow and cancel the siblings on the first error.
# - Each agent enforces its own per-request budget (`agent_timeout`). `_call_agent` adds a
#   per-request backstop (budget + AGENT_GRACE_S) for an agent that ignores it, and the ADK node
#   `timeout=` is a fixed outer guard above the 8 s cap.
# - Node outputs are our dataclasses, passed by reference. This relies on the in-memory session
#   service; a persistent session service would need JSON-serializable outputs.
# - Input validation (400/422) runs before the graph and persistence after it, so the typed
#   errors and the shielded background write keep their current behaviour.
# ---------------------------------------------------------------------------

WORKFLOW_APP_NAME = "verifact_verify"
_WORKFLOW_USER_ID = "verifact-orchestrator"
AGENT_GRACE_S = 0.5                             # per-request backstop beyond an agent's budget
AGENT_NODE_TIMEOUT_S = AGENT_TIMEOUT_CAP + 1.0  # fixed outer guard; never reached in practice


class VerifyRequest(BaseModel):
    """Workflow input (START). `deadline` is a time.monotonic() timestamp."""
    claim: str
    embedding: List[float]
    deadline: float
    bypass_cache: bool = False
    agent_timeout: float = AGENT_TIMEOUT_CAP


@dataclass
class PipelineOutcome:
    """Terminal output of the workflow. Exactly one is emitted per run."""
    cached: bool
    cached_row: Optional[Dict[str, Any]] = None
    fc_res: Optional[factcheck_agent.FactCheckResult] = None
    cr_res: Optional[crossref_agent.CrossRefResult] = None
    ling_res: Optional[linguistic_agent.LinguisticResult] = None
    scoring_res: Optional[ScoringResult] = None
    warnings: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    linguistic_profile: Dict[str, float] = field(default_factory=dict)
    search_suggestions_html: Optional[str] = None
    agents: List[Dict[str, Any]] = field(default_factory=list)


@node(name="cache_lookup")
async def cache_lookup_node(node_input: VerifyRequest) -> Event:
    """Semantic cache check. Fails closed: any error or timeout is a miss."""
    cached = None
    if not node_input.bypass_cache:
        try:
            cached = await asyncio.wait_for(
                lookup_cache(node_input.embedding, claim_text=node_input.claim, similarity_threshold=0.95),
                timeout=CACHE_LOOKUP_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Semantic cache lookup exceeded its budget; treating as a miss.")
        except Exception as exc:
            logger.warning(f"Semantic cache lookup failed; treating as a miss: {exc}")
    if cached:
        return Event(output=cached, route="hit")

    # The agents get whatever the cache lookup left of the request budget.
    timeout = agent_budget(node_input.deadline - time.monotonic())
    return Event(output=node_input.model_copy(update={"agent_timeout": timeout}), route="miss")


@node(name="format_cached")
def format_cached_node(node_input: Dict[str, Any]) -> PipelineOutcome:
    return PipelineOutcome(cached=True, cached_row=node_input)


async def _call_agent(agent_module: Any, result_type: type, req: VerifyRequest) -> Any:
    # Resolved through the module at call time so tests can patch `<agent>.run`.
    res = await asyncio.wait_for(
        agent_module.run(req.claim, timeout=req.agent_timeout),
        timeout=req.agent_timeout + AGENT_GRACE_S,
    )
    if not isinstance(res, result_type):
        raise TypeError(f"{agent_module.__name__} returned {type(res).__name__}, expected {result_type.__name__}")
    return res


async def _factcheck(node_input: VerifyRequest) -> factcheck_agent.FactCheckResult:
    return await _call_agent(factcheck_agent, factcheck_agent.FactCheckResult, node_input)


async def _crossref(node_input: VerifyRequest) -> crossref_agent.CrossRefResult:
    return await _call_agent(crossref_agent, crossref_agent.CrossRefResult, node_input)


async def _linguistic(node_input: VerifyRequest) -> linguistic_agent.LinguisticResult:
    return await _call_agent(linguistic_agent, linguistic_agent.LinguisticResult, node_input)


# (node, provider label for error classification, neutral-result factory), in result order.
AGENT_NODES = (
    (FunctionNode(func=_factcheck, name="factcheck", timeout=AGENT_NODE_TIMEOUT_S),
     "factcheck_agent", factcheck_agent.make_neutral_factcheck),
    (FunctionNode(func=_crossref, name="crossref", timeout=AGENT_NODE_TIMEOUT_S),
     "crossref_agent", crossref_agent.make_neutral_crossref),
    (FunctionNode(func=_linguistic, name="linguistic", timeout=AGENT_NODE_TIMEOUT_S),
     "linguistic_agent", linguistic_agent.make_neutral_linguistic),
)


# rerun_on_resume=True is REQUIRED for a node that calls ctx.run_node (ADK raises ValueError
# otherwise); it is not a HITL setting here.
@node(name="dispatch_agents", rerun_on_resume=True)
async def dispatch_agents_node(ctx: Context, node_input: VerifyRequest) -> Dict[str, Any]:
    """Supervisor: run the three agents concurrently; a failed agent degrades, never the run."""
    async def _timed_run(agent_node):
        t0 = time.monotonic()
        try:
            res = await ctx.run_node(agent_node, node_input)
            dur_ms = int((time.monotonic() - t0) * 1000)
            return res, dur_ms
        except Exception as exc:
            dur_ms = int((time.monotonic() - t0) * 1000)
            return exc, dur_ms

    results = await asyncio.gather(
        *(_timed_run(agent_node) for agent_node, _, _ in AGENT_NODES),
        return_exceptions=True,
    )
    outputs: Dict[str, Any] = {}
    timings: Dict[str, int] = {}
    for (agent_node, provider, make_neutral), item in zip(AGENT_NODES, results):
        if isinstance(item, tuple):
            res, dur_ms = item
        else:
            res, dur_ms = item, 0
        timings[agent_node.name] = dur_ms
        if isinstance(res, BaseException):
            if not isinstance(res, Exception):
                raise res  # cancellation / interrupt: not an agent failure
            # ADK's DynamicNodeFailError wraps the agent's own exception (incl. NodeTimeoutError)
            # as `.error`. Read it duck-typed: the class lives in a private ADK module.
            wrapped = getattr(res, "error", None)
            cause = wrapped if isinstance(wrapped, Exception) else res
            if isinstance(cause, NodeTimeoutError):
                # ADK re-raises ANY TimeoutError leaving a node with `timeout=` as NodeTimeoutError,
                # which is not a TimeoutError subclass; restore it so it classifies as TIMEOUT.
                cause = asyncio.TimeoutError(f"{agent_node.name} agent timed out")
            # ADK has already logged the traceback for the failed child node.
            logger.warning(f"Agent node '{agent_node.name}' degraded: {type(cause).__name__}")
            res = make_neutral(
                degraded=True,
                error=str(cause),
                error_details=classify_provider_error(cause, provider=provider),
            )
        outputs[agent_node.name] = res
    outputs["timings"] = timings
    return outputs


@node(name="score")
def score_node(node_input: Dict[str, Any]) -> PipelineOutcome:
    """Deterministic scoring, capping, warnings and source consolidation."""
    fc_res: factcheck_agent.FactCheckResult = node_input["factcheck"]
    cr_res: crossref_agent.CrossRefResult = node_input["crossref"]
    ling_res: linguistic_agent.LinguisticResult = node_input["linguistic"]
    timings: Dict[str, int] = node_input.get("timings") or {}

    agents_status = []
    for name, res in [("factcheck", fc_res), ("crossref", cr_res), ("linguistic", ling_res)]:
        ed = getattr(res, "error_details", None)
        error_code = ed.code.value if ed else None
        agents_status.append({
            "name": name,
            "status": "degraded" if res.degraded else "ok",
            "duration_ms": timings.get(name),
            "error_code": error_code,
        })

    agents_succeeded = sum(1 for res in (fc_res, cr_res, ling_res) if not res.degraded)
    if agents_succeeded == 0:
        # Guardrail: never score. verify_claim raises AllAgentsFailedError outside the graph.
        return PipelineOutcome(
            cached=False,
            fc_res=fc_res,
            cr_res=cr_res,
            ling_res=ling_res,
            agents=agents_status,
        )

    scoring_res = ScoringEngine().compute(
        s_fact=fc_res.s_fact_score,
        b_ling=ling_res.b_ling_score,
        v_consensus=cr_res.v_consensus_score,
        agents_succeeded=agents_succeeded,
        total_agents=3,
        fact_check_found=fc_res.matched_review,
        verdict_found=fc_res.verdict_found,
        v_refute=cr_res.v_refute_score,
    )

    # Enrich warnings with transparent provider degradation reasons
    enriched_warnings = list(scoring_res.warnings)
    for agent_label, res in [("Fact-check", fc_res), ("Cross-reference", cr_res), ("Linguistic", ling_res)]:
        if res.degraded:
            ed = getattr(res, "error_details", None)
            enriched_warnings.append(degradation_message(agent_label, ed.code.value if ed else None))

    search_suggestions_html = (
        getattr(fc_res, "search_suggestions_html", None)
        or getattr(cr_res, "search_suggestions_html", None)
    )

    return PipelineOutcome(
        cached=False,
        fc_res=fc_res,
        cr_res=cr_res,
        ling_res=ling_res,
        scoring_res=scoring_res,
        warnings=enriched_warnings,
        sources=_consolidate_sources(fc_res, cr_res),
        linguistic_profile={
            "clickbait_syntax": ling_res.clickbait_syntax,
            "emotional_framing": ling_res.emotional_framing,
            "sensationalism": ling_res.sensationalism,
            "informational_density": ling_res.informational_density,
        },
        search_suggestions_html=search_suggestions_html,
        agents=agents_status,
    )


def build_verification_workflow() -> Workflow:
    return Workflow(
        name="verify_claim",
        description="Cache check, parallel fact-check / cross-reference / linguistic agents, scoring.",
        input_schema=VerifyRequest,
        edges=[
            ("START", cache_lookup_node, {"hit": format_cached_node, "miss": dispatch_agents_node}),
            (dispatch_agents_node, score_node),
        ],
    )


_runner: Optional[InMemoryRunner] = None


def get_verification_runner() -> InMemoryRunner:
    """One Runner per process; each request gets its own session."""
    global _runner
    if _runner is None:
        _runner = InMemoryRunner(app=App(name=WORKFLOW_APP_NAME, root_agent=build_verification_workflow()))
    return _runner


async def run_verification_workflow(request: VerifyRequest) -> PipelineOutcome:
    """Drive one workflow run and return its terminal PipelineOutcome."""
    runner = get_verification_runner()
    session_id = f"verify-{uuid.uuid4()}"
    await runner.session_service.create_session(
        app_name=WORKFLOW_APP_NAME, user_id=_WORKFLOW_USER_ID, session_id=session_id
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=request.model_dump_json())])
    outcome: Optional[PipelineOutcome] = None
    try:
        async for event in runner.run_async(user_id=_WORKFLOW_USER_ID, session_id=session_id, new_message=message):
            if isinstance(event.output, PipelineOutcome):
                outcome = event.output
    finally:
        try:
            await runner.session_service.delete_session(
                app_name=WORKFLOW_APP_NAME, user_id=_WORKFLOW_USER_ID, session_id=session_id
            )
        except Exception as cleanup_err:
            logger.debug(f"Failed to delete workflow session {session_id}: {cleanup_err}")
    if outcome is None:
        raise RuntimeError("Verification workflow finished without an outcome")
    return outcome


def _format_cached_response(cached: Dict[str, Any], cleaned_text: str) -> Dict[str, Any]:
    from app.api.result_format import format_stored_result
    return format_stored_result(cached, cleaned_text, cached=True)


def _consolidate_sources(
    fc_res: factcheck_agent.FactCheckResult, cr_res: crossref_agent.CrossRefResult
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []

    # Add fact-check sources
    for m in fc_res.all_matches:
        url_val = m.get("url") or ""
        # Plan 6: a Layer-2 match carries its grounding/fetched domain. Its url may be a Google
        # grounding redirect, whose own domain (google.com) must never pick the bias label.
        domain = m.get("domain") or extract_domain(url_val)
        bias = get_bias_label(domain)
        src_type = "corroborating" if m.get("rating") == "Web Search Result" else "fact_check"
        sources.append({
            "url": url_val,
            "title": m.get("claim_reviewed") or "Fact-Check Match",
            "publisher": m.get("publisher") or "Fact-Checker",
            "rating": m.get("rating"),
            "bias_label": bias,
            "type": src_type,
            "stance": None,
            "found_by": "factcheck",
        })
        
    # Add corroborating sources
    for s in cr_res.sources_found:
        sources.append({
            "url": s.get("url"),
            "title": s.get("title") or "Corroborating Article",
            "publisher": s.get("publisher") or "News Outlet",
            "rating": None,
            "bias_label": s.get("bias_label") or "Unrated",
            "type": "corroborating",
            "stance": s.get("stance"),
            "found_by": "crossref",
        })

    return sources


async def verify_claim(raw_input: str, input_type: str = "text", bypass_cache: bool = False) -> Dict[str, Any]:
    """
    Main orchestration pipeline for claim verification:
    1. Sanitize input (strip HTML/scripts, URL tracking parameters, whitespace).
    2. Validate input constraints (length <= 2000 chars).
    3. Validate claim verifiability (reject subjective opinions & questions -> 422).
    4. Generate the claim embedding.
    5. Run the ADK verification workflow: semantic cache check (cosine >= 0.95, then row policy
       and claim-equivalence checks; skipped when bypass_cache=True); on a miss, the factcheck,
       crossref and linguistic agents in parallel, joined into deterministic scoring + capping.
    6. Guardrail: If all agents fail, raise AllAgentsFailedError (never fabricate a score).
    7. Persist result and claim to Supabase in the background (skipped on bypass_cache).
    8. Return full structured verification response.
    """
    start_time = time.time()
    deadline = time.monotonic() + settings.request_deadline_s

    # 1. Sanitize input
    cleaned_text = sanitize_text(raw_input)

    # 2. Validate length
    valid_len, len_error = validate_input_length(cleaned_text)
    if not valid_len:
        raise InvalidInputError(len_error)

    # 3. Validate claim verifiability
    verifiable, ver_error = is_verifiable_claim(cleaned_text)
    if not verifiable:
        raise NonVerifiableClaimError(ver_error)

    # 4. Generate embedding
    embedding = await asyncio.to_thread(generate_embedding, cleaned_text)

    # 5. Verification workflow
    outcome = await run_verification_workflow(VerifyRequest(
        claim=cleaned_text,
        embedding=embedding,
        deadline=deadline,
        bypass_cache=bypass_cache,
    ))

    if outcome.cached:
        logger.info(f"Returning cached verification result for: {cleaned_text}")
        response = _format_cached_response(outcome.cached_row or {}, cleaned_text)
        response["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return response

    # 6. CRITICAL GUARDRAIL: All agents fail -> return an error, NEVER fabricate a score!
    if outcome.scoring_res is None:
        logger.error(f"All 3 verification agents failed or degraded on claim: {cleaned_text}")
        raise AllAgentsFailedError("All verification agents failed to complete analysis. Unable to verify claim.")

    fc_res, cr_res, ling_res = outcome.fc_res, outcome.cr_res, outcome.ling_res
    scoring_res = outcome.scoring_res
    enriched_warnings = outcome.warnings
    sources = outcome.sources
    linguistic_profile = outcome.linguistic_profile
    verdict_found = fc_res.verdict_found

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 7. Persist result and claim to Supabase (skip on bypass_cache; always persist otherwise so GET /api/results/{id} works; lookup_cache skips low/insufficient)
    claim_id: Optional[str] = None
    if not bypass_cache:
        try:
            persist_task = asyncio.create_task(insert_cache(
                input_type=input_type,
                raw_input=cleaned_text,  # H4: never persist unsanitized input
                extracted_claim=cleaned_text,
                embedding=embedding,
                c_total=scoring_res.c_total,
                s_fact=scoring_res.components.s_fact,
                b_ling=scoring_res.components.b_ling,
                v_consensus=scoring_res.components.v_consensus,
                weights=scoring_res.weights,
                confidence_level=scoring_res.confidence_level,
                agent_outputs={
                    "factcheck": {
                        "matches_found": fc_res.matches_found,
                        "s_fact_score": fc_res.s_fact_score,
                        "degraded": fc_res.degraded,
                        "error": fc_res.error,
                        "error_details": fc_res.error_details.to_dict() if getattr(fc_res, "error_details", None) else None,
                        "duration_ms": next((a["duration_ms"] for a in outcome.agents if a["name"] == "factcheck"), None),
                    },
                    "crossref": {
                        "entities_extracted": cr_res.entities_extracted,
                        "v_consensus_score": cr_res.v_consensus_score,
                        "v_refute_score": cr_res.v_refute_score,
                        "degraded": cr_res.degraded,
                        "error": cr_res.error,
                        "error_details": cr_res.error_details.to_dict() if getattr(cr_res, "error_details", None) else None,
                        "duration_ms": next((a["duration_ms"] for a in outcome.agents if a["name"] == "crossref"), None),
                    },
                    "linguistic": {
                        "b_ling_score": ling_res.b_ling_score,
                        "flagged_phrases": ling_res.flagged_phrases,
                        "degraded": ling_res.degraded,
                        "error": ling_res.error,
                        "error_details": ling_res.error_details.to_dict() if getattr(ling_res, "error_details", None) else None,
                        "duration_ms": next((a["duration_ms"] for a in outcome.agents if a["name"] == "linguistic"), None),
                    },
                    "search_suggestions_html": outcome.search_suggestions_html,
                    "scoring": {
                        "raw_c_total": scoring_res.raw_c_total,
                        "cap_rule": scoring_res.cap_rule,
                    },
                },
                source_citations=sources,
                linguistic_profile=linguistic_profile,
                processing_time_ms=elapsed_ms,
                warnings=enriched_warnings,
                v_refute=scoring_res.components.v_refute,
                evidence_status=scoring_res.evidence_status,
            ))
            # Referenced from creation, so the write survives both the deadline and a
            # cancelled request (client disconnect); released when it finishes.
            _background_tasks.add(persist_task)
            persist_task.add_done_callback(_background_tasks.discard)
            try:
                claim_id = await asyncio.wait_for(
                    asyncio.shield(persist_task),
                    timeout=max(PERSIST_MIN_WAIT_S, deadline - time.monotonic()),
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning("Persistence exceeded the request deadline; returning without an id (write continues).")
                claim_id = None
        except Exception as exc:
            logger.warning(f"Failed to persist result to database: {exc}")

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "id": claim_id,
        "claim": cleaned_text,
        "trust_score": {
            "c_total": scoring_res.c_total,
            "s_fact": scoring_res.components.s_fact,
            "b_ling": scoring_res.components.b_ling,
            "v_consensus": scoring_res.components.v_consensus,
            "v_refute": scoring_res.components.v_refute,
            "weights": scoring_res.weights,
            "raw_c_total": scoring_res.raw_c_total,
            "cap_rule": scoring_res.cap_rule,
        },
        "confidence": {
            "level": scoring_res.confidence_level,
            "warnings": enriched_warnings,
            "verdict_found": verdict_found,
            "evidence_status": scoring_res.evidence_status,
        },
        "linguistic_profile": linguistic_profile,
        "sources": sources,
        "processing_time_ms": elapsed_ms,
        "cached": False,
        "search_suggestions_html": outcome.search_suggestions_html,
        "agents": outcome.agents,
        "created_at": created_at,
    }
