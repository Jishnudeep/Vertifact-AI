"""
Eval-only record/replay for external calls, plus leave-source-out filtering (C3, Plan 6).

Never imported by the backend. run_eval.py installs it around a run:
  record: call live, store successful responses under cache_dir
  replay: serve from cache_dir only; a missing entry raises ReplayMissError (an outage)
  live:   call live, store nothing
Cached payloads are RAW results. Domain exclusion is applied on the way out, so one cache
serves runs with and without leave-source-out.
"""
import contextlib
import contextvars
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import app.agents.crossref_agent as crossref_agent
import app.agents.factcheck_agent as factcheck_agent
import app.agents.linguistic_agent as linguistic_agent
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
        async def grounded_search(kind: SearchKind, claim: str):
            payload = {"kind": kind, "claim": claim, "model": settings.search_model_id}
            cached = await self._through_cache(
                "gemini_search", payload,
                lambda: original(kind=kind, claim=claim),
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

    def _wrap_run_adk_agent(self, original):
        async def run_adk_agent(name, instruction, prompt, app_name):
            payload = {"model": settings.model_id, "name": name, "instruction": instruction,
                       "prompt": prompt, "app_name": app_name}
            return await self._through_cache(
                "llm", payload, lambda: original(name, instruction, prompt, app_name)
            )
        return run_adk_agent

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
        self._patch(factcheck_agent, "run_adk_agent", self._wrap_run_adk_agent)
        self._patch(linguistic_agent, "run_adk_agent", self._wrap_run_adk_agent)

    def uninstall(self) -> None:
        while self._originals:
            module, attr, original = self._originals.pop()
            setattr(module, attr, original)
