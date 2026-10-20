import sys
from pathlib import Path
import os
import pytest

pytest_plugins = ["pytester"]

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
os.environ.setdefault("GOOGLE_FACT_CHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GOOGLE_FACTCHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GEMINI_API_KEY", "dummy_gemini_key")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run tests marked with live_api that make external API calls",
    )


def pytest_configure(config):
    # eval/tests is collected with the repo root as rootdir, so backend/pyproject.toml's marker
    # registration does not apply here. Live tests in this suite use only the local NLI model.
    config.addinivalue_line("markers", "live: tests that need a real model or external API")
    config.addinivalue_line("markers", "live_api: tests that call external APIs and spend quota (opt-in)")


def pytest_collection_modifyitems(config, items):
    run_live_api = config.getoption("--run-live-api") or os.environ.get("VERIFACT_RUN_LIVE_API") == "1"
    if not run_live_api:
        skip_live_api = pytest.mark.skip(reason="need --run-live-api option or VERIFACT_RUN_LIVE_API=1 to run")
        for item in items:
            if "live_api" in item.keywords:
                item.add_marker(skip_live_api)


@pytest.fixture
def anyio_backend():
    return "asyncio"
