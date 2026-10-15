import os
import pytest

# Ensure dummy environment variables exist for offline unit testing
# before any application modules are imported.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_PUBLISH_KEY", "dummy_publish_key")
os.environ.setdefault("SUPABASE_SECRET_KEY", "dummy_secret_key")
os.environ.setdefault("GROQ_API_KEY", "dummy_groq_key")
os.environ.setdefault("TAVILY_KEY", "dummy_tavily_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_tavily_key")
os.environ.setdefault("GOOGLE_FACT_CHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GOOGLE_FACTCHECK_API_KEY", "dummy_factcheck_key")
# Plan 6: a dummy Gemini key wins over the real one in app/.env (env beats dotenv), so the real
# key never reaches the unit-test process.
os.environ.setdefault("GEMINI_API_KEY", "dummy_gemini_key")

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def _fresh_budgets():
    from app.services import budget
    budget.get_tavily_budget.cache_clear()
    budget.get_llm_budget.cache_clear()
    budget.get_search_budget.cache_clear()
    yield


_LIVE_SEARCH_MSG = "live Gemini search called in a unit test; patch grounded_search or mark live_search"


@pytest.fixture(autouse=True)
def _no_live_grounded_search(request, monkeypatch):
    """
    Plan 6: block real Gemini grounded-search calls in unit tests. A test opts out with
    `@pytest.mark.live_search`, or simply patches `run_grounded_agent` itself (its patch is
    applied after this one and wins).

    The stub raises pytest.fail (a BaseException) so `except Exception` outage handling can't
    absorb it, and teardown fails the test as well, because `gather(return_exceptions=True)`
    would otherwise hand the exception back as a value and let the test pass.
    """
    calls = []

    async def _blocked(*_args, **_kwargs):
        calls.append(1)
        pytest.fail(_LIVE_SEARCH_MSG)

    if request.node.get_closest_marker("live_search") is None:
        from app.services import grounded_search
        from app.utils import adk_helpers
        monkeypatch.setattr(adk_helpers, "run_grounded_agent", _blocked)
        monkeypatch.setattr(grounded_search, "run_grounded_agent", _blocked)
    yield
    if calls:
        pytest.fail(_LIVE_SEARCH_MSG)

