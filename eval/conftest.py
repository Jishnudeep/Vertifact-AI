import sys
from pathlib import Path
import os
import pytest

# Ensure the repo ROOT is importable (so `import eval.metrics...` resolves) and the backend
# directory too (so `import app...` resolves). Without the root entry,
# `pytest eval/tests` from the repo root fails to collect with ModuleNotFoundError: 'eval'.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

backend_path = repo_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Ensure dummy environment variables for offline eval tests
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_PUBLISH_KEY", "dummy_publish_key")
os.environ.setdefault("SUPABASE_SECRET_KEY", "dummy_secret_key")
os.environ.setdefault("GROQ_API_KEY", "dummy_groq_key")
os.environ.setdefault("TAVILY_KEY", "dummy_tavily_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_tavily_key")
os.environ.setdefault("GOOGLE_FACT_CHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GOOGLE_FACTCHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GEMINI_API_KEY", "dummy_gemini_key")

def pytest_configure(config):
    # eval/tests is collected with the repo root as rootdir, so backend/pyproject.toml's marker
    # registration does not apply here. Live tests in this suite use only the local NLI model.
    config.addinivalue_line("markers", "live: tests that need a real model or external API")


@pytest.fixture
def anyio_backend():
    return "asyncio"
