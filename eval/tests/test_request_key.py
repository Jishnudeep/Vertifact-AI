"""request_key must change whenever anything that changes the model's answer changes (Task C, N10)."""
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from eval.replay import request_key


class SchemaA(BaseModel):
    answer: str


class SchemaB(BaseModel):
    answer: str
    confidence: float


def _request(**config):
    return LlmRequest(
        model="m",
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=types.GenerateContentConfig(system_instruction="sys", **config),
    )


def test_identical_requests_share_a_key():
    assert request_key(_request(temperature=0.0)) == request_key(_request(temperature=0.0))


def test_a_different_response_schema_is_a_different_key():
    assert request_key(_request(response_schema=SchemaA)) != request_key(_request(response_schema=SchemaB))
    assert request_key(_request(response_schema=SchemaA)) != request_key(_request())


def test_generation_settings_are_part_of_the_key():
    base = request_key(_request())
    assert request_key(_request(temperature=0.7)) != base
    assert request_key(_request(max_output_tokens=64)) != base
    assert request_key(_request(thinking_config=types.ThinkingConfig(thinking_budget=0))) != base


import pytest

from eval.replay import EvidenceReplay, ReplayMissError
import app.agents.factcheck_agent as factcheck_tools


@pytest.mark.anyio
async def test_a_miss_is_visible_on_the_claim_that_hit_it(tmp_path):
    replay = EvidenceReplay(tmp_path, "replay")
    replay.install()
    try:
        replay.reset_claim_counts("c1")
        with pytest.raises(ReplayMissError):
            await factcheck_tools.grounded_search("factcheck", "unrecorded")
        assert replay.claim_misses() == 1 and replay.misses == 1
        replay.reset_claim_counts("c2")
        assert replay.claim_misses() == 0 and replay.misses == 1
    finally:
        replay.uninstall()
