"""
Eval-only record/replay for external calls, plus leave-source-out filtering (C3, Plan 6).

Never imported by the backend. run_eval.py installs it around a run:
  record: call live, store successful responses under cache_dir
  replay: serve from cache_dir only; a missing entry raises ReplayMissError (an outage)
  live:   call live, store nothing
Cached search and fact-check payloads are RAW results; domain exclusion is applied on the way out.
LLM turns are keyed on what the model was shown, so a leave-source-out run needs its own recording.

Two seams are patched, both where the worker agents look the dependency up:
  * the services the agents' TOOLS call (`app.agents.factcheck_agent` / `crossref_agent`:
    grounded_search, fetch_page_text, search_factchecks), and
  * the single model seam `app.agents.common.get_model`, which is wrapped in a `ReplayLlm`
    keyed on a hash of the serialized LlmRequest (system instruction, contents, tool declarations).
Replay is deterministic only as long as the agents' prompts and tool declarations do not change; a
changed prompt is a new key, i.e. a replay miss that invalidates the run (see run_eval).
"""
import contextlib
import contextvars
from dataclasses import asdict
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from pydantic import ConfigDict

import app.agents.common as common
import app.agents.crossref_agent as crossref_agent
import app.agents.factcheck_agent as factcheck_agent
from app.config import settings
from app.services.errors import EvidenceAPIError
from app.services.grounded_search import GroundedSearch, GroundedSource, SearchKind
from app.services.page_fetch import PageText
from app.utils.text_processing import extract_domain

logger = logging.getLogger(__name__)

ReplayMode = Literal["record", "replay", "live", "record-missing"]

_excluded: contextvars.ContextVar[frozenset] = contextvars.ContextVar("eval_excluded_domains", default=frozenset())


class ReplayMissError(EvidenceAPIError):
    """No recorded response for this call in replay mode. Agents treat it as an outage."""


@contextlib.contextmanager
def excluding_domains(domains: Iterable[str]):
    token = _excluded.set(frozenset(d.lower() for d in domains if d))
    try:
        yield
    finally:
        _excluded.reset(token)


def filter_web_results(results: List[Dict[str, Any]], excluded: frozenset) -> List[Dict[str, Any]]:
    return [r for r in results if extract_domain(r.get("url", "")) not in excluded]


def filter_factcheck_claims(claims: List[Dict[str, Any]], excluded: frozenset) -> List[Dict[str, Any]]:
    kept = []
    for claim in claims:
        reviews = [rv for rv in claim.get("claimReview", []) if extract_domain(rv.get("url", "")) not in excluded]
        if reviews:
            kept.append({**claim, "claimReview": reviews})
    return kept


def _key(provider: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps([provider, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EvidenceReplay:
    def __init__(self, cache_dir: Path, mode: ReplayMode):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.misses = 0
        self._current_claim_misses = 0
        self.recorded = 0
        self._originals: List[Tuple[Any, str, Callable]] = []
        self.calls_by_provider: Dict[str, int] = {
            "search": 0,
            "fetch": 0,
            "factcheck_api": 0,
            "llm": 0,
        }
        self._current_claim_calls: Dict[str, int] = {
            "search": 0,
            "fetch": 0,
            "factcheck_api": 0,
            "llm": 0,
        }
        self.leaf_transcript: List[Dict[str, Any]] = []
        self.current_claim_id: Optional[str] = None

    def write_leaf_transcript(self, path: Path) -> None:
        """One JSON object per model turn, each tagged with the claim it belongs to."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.leaf_transcript:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def note_miss(self) -> None:
        self.misses += 1
        self._current_claim_misses += 1

    def claim_misses(self) -> int:
        """Replay misses during the current claim: a claim can look normal while one of its calls missed."""
        return self._current_claim_misses

    def reset_claim_counts(self, claim_id: Optional[str] = None) -> None:
        self.current_claim_id = claim_id
        self._current_claim_misses = 0
        self._current_claim_calls = {
            "search": 0,
            "fetch": 0,
            "factcheck_api": 0,
            "llm": 0,
        }

    def claim_counts(self) -> Dict[str, int]:
        return dict(self._current_claim_calls)

    def _record_provider_call(self, provider_alias: str) -> None:
        if provider_alias in self.calls_by_provider:
            self.calls_by_provider[provider_alias] += 1
            self._current_claim_calls[provider_alias] += 1

    # --- cache plumbing -------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def _through_cache(self, provider: str, payload: Dict[str, Any], live_call):
        key = _key(provider, payload)
        path = self._path(key)
        provider_alias_map = {
            "gemini_search": "search",
            "page_fetch": "fetch",
            "google_factcheck": "factcheck_api",
            "llm": "llm",
        }
        alias = provider_alias_map.get(provider, provider)

        if self.mode == "replay" or (self.mode == "record-missing" and path.exists()):
            if not path.exists():
                self.note_miss()
                raise ReplayMissError(f"No recorded {provider} response (key {key[:12]})", provider=provider)
            res = json.loads(path.read_text(encoding="utf-8"))
            self._record_provider_call(alias)
            return res

        result = await live_call()  # errors propagate and are never recorded
        self._record_provider_call(alias)
        if self.mode in ("record", "record-missing"):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(result, "to_dict"):
                serializable = result.to_dict()
            elif result is not None and hasattr(result, "__dataclass_fields__"):
                serializable = asdict(result)
            else:
                serializable = result
            path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
            self.recorded += 1
        return result

    # --- wrappers ---------------------------------------------------------
    def _wrap_grounded_search(self, original):
        async def grounded_search(kind: SearchKind, query: str):
            # The payload key stays "claim": entries recorded before the tool-calling refactor
            # (where the query WAS the claim) keep hitting.
            payload = {"kind": kind, "claim": query, "model": settings.search_model_id}
            cached = await self._through_cache(
                "gemini_search", payload,
                lambda: original(kind, query),
            )
            gs = cached if isinstance(cached, GroundedSearch) else GroundedSearch.from_dict(cached)
            excluded = _excluded.get()
            if excluded:
                filtered_sources = [s for s in gs.sources if s.domain.lower() not in excluded]
                return GroundedSearch(
                    sources=filtered_sources,
                    queries=gs.queries,
                    search_suggestions_html=gs.search_suggestions_html,
                )
            return gs
        return grounded_search

    def _wrap_fetch_page_text(self, original):
        async def fetch_page_text(url: str, expected_domain: Optional[str] = None):
            payload = {"url": url, "expected_domain": expected_domain}
            cached = await self._through_cache(
                "page_fetch", payload,
                lambda: original(url=url, expected_domain=expected_domain),
            )
            if cached is None:
                return None
            if isinstance(cached, PageText):
                return cached
            return PageText(**cached)
        return fetch_page_text

    def _wrap_search_factchecks(self, original):
        async def search_factchecks(query, api_key):
            claims = await self._through_cache(
                "google_factcheck", {"query": query}, lambda: original(query, api_key)
            )
            return filter_factcheck_claims(claims, _excluded.get())
        return search_factchecks

    def _wrap_agent_kwargs(self, original):
        def agent_kwargs() -> Dict[str, Any]:
            kwargs = dict(original())
            before = kwargs.get("before_model_callback") or []
            before = list(before) if isinstance(before, (list, tuple)) else [before]
            kwargs["before_model_callback"] = [_note_agent, *before]
            return kwargs
        return agent_kwargs

    def _wrap_get_model(self, original):
        def get_model() -> BaseLlm:
            inner = original()
            if isinstance(inner, str):
                inner = Gemini(model=inner)
            return ReplayLlm(model=inner.model, inner=inner, replay=self)
        return get_model

    # --- install / uninstall ----------------------------------------------
    def _patch(self, module, attr: str, wrapper_factory) -> None:
        original = getattr(module, attr)
        self._originals.append((module, attr, original))
        setattr(module, attr, wrapper_factory(original))

    def install(self) -> None:
        if self.mode == "live":
            return
        self._patch(crossref_agent, "grounded_search", self._wrap_grounded_search)
        self._patch(factcheck_agent, "grounded_search", self._wrap_grounded_search)
        self._patch(crossref_agent, "fetch_page_text", self._wrap_fetch_page_text)
        self._patch(factcheck_agent, "fetch_page_text", self._wrap_fetch_page_text)
        self._patch(factcheck_agent, "search_factchecks", self._wrap_search_factchecks)
        self._patch(common, "get_model", self._wrap_get_model)
        self._patch(common, "agent_kwargs", self._wrap_agent_kwargs)

    def uninstall(self) -> None:
        while self._originals:
            module, attr, original = self._originals.pop()
            setattr(module, attr, original)


# --- model-level replay -------------------------------------------------------------------------

def _without_ids(node):
    """Function call/response ids are random per run; they must not reach the replay key."""
    if isinstance(node, dict):
        return {k: _without_ids(v) for k, v in node.items()
                if not (k == "id" and ("name" in node) and ("args" in node or "response" in node))}
    if isinstance(node, list):
        return [_without_ids(v) for v in node]
    return node


_GENERATION_FIELDS = ("temperature", "top_p", "top_k", "max_output_tokens", "candidate_count", "seed",
                      "response_mime_type", "presence_penalty", "frequency_penalty")


def _generation_settings(config: Any) -> Dict[str, Any]:
    """What shapes the answer besides the prompt: sampling, thinking budget and the output schema."""
    out: Dict[str, Any] = {f: getattr(config, f, None) for f in _GENERATION_FIELDS}
    thinking = getattr(config, "thinking_config", None)
    out["thinking_config"] = thinking.model_dump(mode="json", exclude_none=True) if thinking else None
    schema = getattr(config, "response_schema", None)
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        out["response_schema"] = schema.model_json_schema()
    elif hasattr(schema, "model_dump"):
        out["response_schema"] = schema.model_dump(mode="json", exclude_none=True)
    else:
        out["response_schema"] = schema
    return {k: v for k, v in out.items() if v is not None}


_DATE_SENTENCE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2} \(UTC\)\.\s*")


def request_key(llm_request: LlmRequest) -> str:
    """Hash of what the model is actually shown: system instruction, contents, tool declarations, generation settings."""
    config = llm_request.config
    system = getattr(config, "system_instruction", None)
    if isinstance(system, str):
        # Normalize dynamic date injection (Task A.2) so replay cache keys remain stable across days
        system = _DATE_SENTENCE.sub("", system)
    elif system is not None and hasattr(system, "model_dump"):
        dumped = system.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict) and "parts" in dumped:
            for part in dumped.get("parts", []):
                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                    part["text"] = _DATE_SENTENCE.sub("", part["text"])
        system = dumped
    canonical = {
        "generation": _generation_settings(config),
        "model": llm_request.model,
        "system_instruction": system,
        "contents": [_without_ids(c.model_dump(mode="json", exclude_none=True)) for c in llm_request.contents],
        "tools": [t.model_dump(mode="json", exclude_none=True) for t in (getattr(config, "tools", None) or [])],
    }
    return _key("llm", canonical)


# The agent whose model turn is being served. Workers run through ctx.run_node, whose requests carry no
# "internal name is ..." line, so the name is taken from ADK's callback context (the documented
# `common.agent_kwargs` seam) in the same task that then calls the model.
_current_agent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("eval_current_agent", default=None)
_INTERNAL_NAME = re.compile(r'internal name is "([\w-]+)"')


def _note_agent(callback_context, llm_request=None):
    """before_model_callback: remember which agent is about to call the model."""
    _current_agent.set(getattr(callback_context, "agent_name", None))
    return None


def _agent_name(system_instruction: str, fallback: str) -> str:
    """The calling agent's ADK name; then the identity line ADK adds to some requests; then the model id."""
    current = _current_agent.get()
    if current:
        return current
    match = _INTERNAL_NAME.search(system_instruction)
    return match.group(1) if match else fallback


class ReplayLlm(BaseLlm):
    """
    Wraps the real worker model (`get_model()`) and serves/records each turn's LlmResponses by
    `request_key`. In replay mode the inner model is never called; a missing entry raises
    ReplayMissError (an outage: the agent degrades and the run is invalidated).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "replay"
    inner: object
    replay: object

    @property
    def capabilities(self) -> LlmCapabilities:
        return self.inner.capabilities                      # same structured-output path as the real model

    def _record_leaf_transcript(self, llm_request: LlmRequest, responses: List[LlmResponse]) -> None:
        """
        One structured entry per model turn: the agent (ADK's own "internal name" line in the system
        instruction), the claim being evaluated, every tool response the model has been shown so far,
        this turn's tool calls and, on the final turn, the structured decision.
        """
        replay: EvidenceReplay = self.replay  # type: ignore[assignment]
        config = getattr(llm_request, "config", None)
        system = getattr(config, "system_instruction", "") or ""
        if isinstance(system, str):
            sys_str = system
        elif hasattr(system, "parts"):
            sys_str = " ".join(getattr(p, "text", "") or "" for p in system.parts)
        else:
            sys_str = str(system)

        prompt_parts: List[str] = []
        tool_responses: List[Dict[str, Any]] = []
        for c in (llm_request.contents or []):
            for p in (getattr(c, "parts", []) or []):
                if getattr(p, "text", None):
                    prompt_parts.append(p.text)
                elif getattr(p, "function_response", None):
                    fr = p.function_response
                    tool_responses.append({"name": fr.name, "response": fr.response})
                    prompt_parts.append(json.dumps(fr.response))
        prompt_text = "\n".join(prompt_parts)

        resp_texts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        decision: Optional[Dict[str, Any]] = None
        for r in responses:
            for p in (getattr(getattr(r, "content", None), "parts", []) or []):
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    args = dict(fc.args or {})
                    if fc.name == "set_model_response":
                        decision = args
                    else:
                        tool_calls.append({"name": fc.name, "args": args})
                    resp_texts.append(f"call:{fc.name}({json.dumps(args)})")
                elif getattr(p, "text", None):
                    resp_texts.append(p.text)
        combined_resp = "\n".join(resp_texts)
        try:
            resp_payload = json.loads(combined_resp)
        except Exception:
            resp_payload = combined_resp
        if decision is None and isinstance(resp_payload, dict) and not tool_calls:
            decision = resp_payload                     # Gemini-native structured output: the final text is the JSON

        replay.leaf_transcript.append({
            "claim_id": replay.current_claim_id,
            "name": _agent_name(sys_str, getattr(self.inner, "model", "llm_agent")),
            "prompt": prompt_text,
            "response": resp_payload,
            "tool_calls": tool_calls,
            "tool_responses": tool_responses,
            "decision": decision,
        })

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        replay: EvidenceReplay = self.replay                # type: ignore[assignment]
        key = request_key(llm_request)
        path = replay._path(key)
        if replay.mode == "replay" or (replay.mode == "record-missing" and path.exists()):
            if not path.exists():
                replay.note_miss()
                raise ReplayMissError(f"No recorded llm response (key {key[:12]})", provider="llm")
            responses: List[LlmResponse] = []
            for item in json.loads(path.read_text(encoding="utf-8")):
                r = LlmResponse.model_validate(item)
                responses.append(r)
                yield r
            replay._record_provider_call("llm")
            self._safe_record_transcript(llm_request, responses)
            return

        responses = []
        async for response in self.inner.generate_content_async(llm_request, stream=stream):
            responses.append(response)
            yield response
        replay._record_provider_call("llm")
        if replay.mode in ("record", "record-missing"):
            replay.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps([r.model_dump(mode="json", exclude_none=True) for r in responses], ensure_ascii=False),
                encoding="utf-8",
            )
            replay.recorded += 1
        self._safe_record_transcript(llm_request, responses)

    def _safe_record_transcript(self, llm_request: LlmRequest, responses: List[LlmResponse]) -> None:
        """The transcript is diagnostics: a bug in it must never drop a paid recording or fail the turn."""
        try:
            self._record_leaf_transcript(llm_request, responses)
        except Exception:  # noqa: BLE001
            logger.exception("leaf transcript entry skipped")
