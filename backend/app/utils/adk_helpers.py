"""
Grounded-search ADK helper (Plan 6).

The worker agents are real ADK agents now (`app/agents/`), with their own inner `LlmAgent`s,
tools and callbacks. What stays here is the Gemini `google_search` sub-agent that sits BEHIND the
search function tools: `google_search` is a Gemini-only built-in that cannot be mixed with function
tools, so it is wrapped as a service (`app/services/grounded_search.py`)
that the tools call.
"""
import logging
import re
import uuid
from typing import Any, Dict, Literal, Optional

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# Plan 6: grounded-search agents, one cached InMemoryRunner per (kind, model, instruction)
SearchKind = Literal["factcheck", "coverage"]
_grounded_runner_cache: Dict[str, InMemoryRunner] = {}


class GroundedSearchBlocked(RuntimeError):
    """
    A grounded-search turn ended in an ADK error event (blocked / failed generation) with no
    grounding metadata. Carries only a sanitized code name — never the provider's
    `error_message`, which may echo the claim.
    """

    def __init__(self, error_code: Any):
        raw = getattr(error_code, "name", None) or str(error_code)
        self.code = re.sub(r"[^A-Za-z0-9_]", "", raw)[:64] or "UNKNOWN"
        super().__init__(f"Grounded search returned error event {self.code}")

def get_grounded_runner(kind: SearchKind, instruction: str) -> InMemoryRunner:
    """
    Retrieve the cached InMemoryRunner for a grounded-search agent (Plan 6 D2).

    The agent is a plain `LlmAgent` holding ADK's built-in `google_search` tool, which only
    runs on Gemini models — so the model is `settings.search_model_id` (same Gemini id as
    `settings.model_id`). google-genai reads the Gemini key from GOOGLE_API_KEY, exported by
    `app.config` (Plan 6 D1).
    """
    model_id = settings.search_model_id
    cache_key = f"{kind}:{model_id}:{instruction}"
    runner = _grounded_runner_cache.get(cache_key)
    if runner is None:
        logger.info(f"Creating grounded-search ADK runner for kind '{kind}' (model: {model_id})")
        agent = LlmAgent(
            name=f"{kind}_search",
            model=model_id,
            description=f"Grounded Google Search agent ({kind})",
            instruction=instruction,
            tools=[google_search],
        )
        runner = InMemoryRunner(agent=agent, app_name=f"verifact_{kind}_search")
        _grounded_runner_cache[cache_key] = runner
    return runner

async def run_grounded_agent(
    kind: SearchKind, instruction: str, prompt: str
) -> Optional[types.GroundingMetadata]:
    """
    Run one grounded-search turn and return the grounding metadata it cited.

    Plan 6 D4: the grounded ANSWER TEXT is discarded here and never leaves this function —
    only `grounding_metadata` (cited chunks, queries, search-suggestion HTML) is returned.
    The last event carrying metadata wins; None means the model cited nothing.

    ADK turns a blocked or failed generation (SAFETY, RECITATION, BLOCKLIST,
    PROHIBITED_CONTENT, SPII, OTHER, MAX_TOKENS without text, a prompt block_reason, ...) into
    an event with `error_code` set and the grounding metadata dropped. When no event carried
    metadata and any event carried an `error_code`, this raises `GroundedSearchBlocked`: the
    search could not run, which is an outage, not an absence of evidence.

    Does NOT charge the worker LLM budget (the caller charges the search budget). Provider
    errors (google.genai.errors.APIError, etc.) and cancellation propagate unchanged; the
    session is always deleted.
    """
    runner = get_grounded_runner(kind, instruction)
    session_service = runner.session_service
    app_name = runner.app_name

    run_id = uuid.uuid4().hex[:8]
    user_id = f"{app_name}-user-{run_id}"
    session_id = f"{app_name}-session-{run_id}"
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    metadata: Optional[types.GroundingMetadata] = None
    error_code: Any = None
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
            event_metadata = getattr(event, "grounding_metadata", None)
            if event_metadata is not None:
                metadata = event_metadata
            event_error = getattr(event, "error_code", None)
            if event_error is not None:
                error_code = event_error
        if metadata is None and error_code is not None:
            raise GroundedSearchBlocked(error_code)
        return metadata
    finally:
        try:
            await session_service.delete_session(app_name=app_name, user_id=user_id, session_id=session_id)
        except Exception as cleanup_err:
            logger.debug(f"Failed to delete grounded-search session {session_id}: {cleanup_err}")
