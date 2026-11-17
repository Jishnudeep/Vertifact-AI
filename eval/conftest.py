import socket
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
os.environ["SUPABASE_URL"] = "https://example.supabase.co"   # never a developer's real project
os.environ["SUPABASE_PUBLISH_KEY"] = "dummy_publish_key"
os.environ["SUPABASE_SECRET_KEY"] = "dummy_secret_key"
os.environ.setdefault("GOOGLE_FACT_CHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GOOGLE_FACTCHECK_API_KEY", "dummy_factcheck_key")
os.environ.setdefault("GEMINI_API_KEY", "dummy_gemini_key")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# Offline suite: litellm must not fetch its cost map and Hugging Face must not check the hub.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


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


class NetworkBlocked(ConnectionError):
    """An offline eval test tried to reach a non-loopback host."""


class NetworkGuard:
    def __init__(self):
        self.attempts = []

    @staticmethod
    def _is_local(host) -> bool:
        return host in (None, "", "localhost", "127.0.0.1", "::1", "0.0.0.0") or str(host).startswith("127.")

    def check(self, host) -> None:
        if not self._is_local(host):
            self.attempts.append(str(host))
            raise NetworkBlocked(f"offline eval test tried to reach {host!r}")


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """`pytest eval/tests` must make no network calls: block non-loopback DNS and connects, fail on any attempt."""
    guard = NetworkGuard()
    if "live_api" in request.keywords:
        yield guard
        return

    real_getaddrinfo, real_connect, real_connect_ex = socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        guard.check(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        guard.check(address[0] if isinstance(address, tuple) else address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard.check(address[0] if isinstance(address, tuple) else address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield guard
    assert not guard.attempts, f"network attempted in an offline test: {sorted(set(guard.attempts))}"


@pytest.fixture(autouse=True)
def bias_ratings_from_local_files(monkeypatch):
    """AllSides ratings come from the local CSV in tests, never from a Supabase round-trip."""
    import app.services.bias_lookup as bias_lookup

    def offline():
        raise RuntimeError("offline eval test: Supabase is not reachable")

    monkeypatch.setattr(bias_lookup, "get_supabase", offline)
