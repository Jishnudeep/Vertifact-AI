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

