from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agents.orchestrator import AllAgentsFailedError, InvalidInputError, NonVerifiableClaimError
from app.api.routes import limiter
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    # /api/verify allows 10/min per client; every TestClient request shares one key.
    limiter.reset()
    yield


def _post(body):
    return client.post("/api/verify", json=body)


def test_opinion_gets_non_verifiable_code():
    with patch("app.api.routes.verify_claim", side_effect=NonVerifiableClaimError("opinion")):
        r = _post({"input": "I think pizza is great"})
    assert r.status_code == 422 and r.json()["code"] == "non_verifiable_claim"


def test_oversize_body_gets_request_validation_code():
    r = _post({"input": "x" * 10001})
    assert r.status_code == 422 and r.json()["code"] == "request_validation"


def test_missing_field_gets_request_validation_code():
    r = _post({})
    assert r.status_code == 422 and r.json()["code"] == "request_validation"


def test_too_long_claim_gets_invalid_input_code():
    with patch("app.api.routes.verify_claim", side_effect=InvalidInputError("too long")):
        r = _post({"input": "a claim"})
    assert r.status_code == 400 and r.json()["code"] == "invalid_input"


def test_all_agents_failed_code():
    with patch("app.api.routes.verify_claim", side_effect=AllAgentsFailedError("down")):
        r = _post({"input": "a claim"})
    assert r.status_code == 500 and r.json()["code"] == "all_agents_failed"
