"""
Eval-only record/replay for external calls, plus leave-source-out filtering (C3, Plan 6).

Never imported by the backend. run_eval.py installs it around a run:
  record: call live, store successful responses under cache_dir
  replay: serve from cache_dir only; a missing entry raises ReplayMissError (an outage)
  live:   call live, store nothing
Cached payloads are RAW results. Domain exclusion is applied on the way out, so one cache
serves runs with and without leave-source-out.

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

ReplayMode = Literal["record", "replay", "live"]

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
        self.recorded = 0
        self._originals: List[Tuple[Any, str, Callable]] = []

    # --- cache plumbing -------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def _through_cache(self, provider: str, payload: Dict[str, Any], live_call):
        key = _key(provider, payload)
        path = self._path(key)
        if self.mode == "replay":
            if not path.exists():
                self.misses += 1
                raise ReplayMissError(f"No recorded {provider} response (key {key[:12]})", provider=provider)
            return json.loads(path.read_text(encoding="utf-8"))
        result = await live_call()  # errors propagate and are never recorded
        if self.mode == "record":
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
        self._patch(common, "get_today_utc", lambda _orig: (lambda: "2026-10-20"))

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


def request_key(llm_request: LlmRequest) -> str:
    """Hash of what the model is actually shown: system instruction, contents, tool declarations."""
    config = llm_request.config
    system = getattr(config, "system_instruction", None)
    if isinstance(system, str):
        # Normalize dynamic date injection (Task A.2) so replay cache keys remain stable across days
        system = re.sub(r"Today's date is \d{4}-\d{2}-\d{2} \(UTC\)\.\s*", "", system)
    elif system is not None and hasattr(system, "model_dump"):
        dumped = system.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict) and "parts" in dumped:
            for part in dumped.get("parts", []):
                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                    part["text"] = re.sub(r"Today's date is \d{4}-\d{2}-\d{2} \(UTC\)\.\s*", "", part["text"])
        system = dumped
    canonical = {
        "model": llm_request.model,
        "system_instruction": system,
        "contents": [_without_ids(c.model_dump(mode="json", exclude_none=True)) for c in llm_request.contents],
        "tools": [t.model_dump(mode="json", exclude_none=True) for t in (getattr(config, "tools", None) or [])],
    }
    return _key("llm", canonical)


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

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        replay: EvidenceReplay = self.replay                # type: ignore[assignment]
        key = request_key(llm_request)
        path = replay._path(key)
        if replay.mode == "replay":
            if not path.exists():
                replay.misses += 1
                raise ReplayMissError(f"No recorded llm response (key {key[:12]})", provider="llm")
            for item in json.loads(path.read_text(encoding="utf-8")):
                yield LlmResponse.model_validate(item)
            return

        responses: List[LlmResponse] = []
        async for response in self.inner.generate_content_async(llm_request, stream=stream):
            responses.append(response)
            yield response
        if replay.mode == "record":
            replay.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps([r.model_dump(mode="json", exclude_none=True) for r in responses], ensure_ascii=False),
                encoding="utf-8",
            )
            replay.recorded += 1
