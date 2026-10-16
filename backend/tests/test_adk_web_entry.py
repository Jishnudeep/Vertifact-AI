"""The `adk web` entry point (backend/verifact): model warm-up guard and an end-to-end smoke run.

adk web has no startup hook, so the wrapper must warm the models itself, before the request
deadline starts, and must keep every input guardrail the API path has.
"""
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agents.crossref_agent import CrossRefResult
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult
from app.agents.orchestrator import InvalidInputError, NonVerifiableClaimError, PipelineOutcome
from app.services import warmup
from verifact.agent import root_agent

CLAIM = "The Earth orbits the Sun"


@pytest.fixture
def cold_warmup(monkeypatch):
    monkeypatch.setattr(warmup, "_warmed", False)


def test_ensure_models_warm_loads_once(cold_warmup):
    with patch.object(warmup, "warm_models") as warm:
        warmup.ensure_models_warm()
        warmup.ensure_models_warm()
    warm.assert_called_once_with()


def test_ensure_models_warm_retries_after_a_failed_load(cold_warmup):
    with patch.object(warmup, "warm_models", side_effect=[RuntimeError("spacy missing"), None]) as warm:
        with pytest.raises(RuntimeError):
            warmup.ensure_models_warm()
        warmup.ensure_models_warm()
    assert warm.call_count == 2


def _agent_results():
    fc = make_neutral_factcheck(degraded=False)
    cr = CrossRefResult(entities_extracted=[], sources_found=[], coverage_distribution={
        "Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0, degraded=False)
    ling = LinguisticResult(clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
                            informational_density=1.0, b_ling_score=0.0, degraded=False)
    return fc, cr, ling


async def _run(text: str, calls: list, embed: MagicMock) -> list:
    """Send {"text": text} to root_agent as the dev UI does and return the events."""
    fc, cr, ling = _agent_results()
    with ExitStack() as stack:
        stack.enter_context(patch("verifact.agent.ensure_models_warm", side_effect=lambda: calls.append("warm")))
        stack.enter_context(patch("app.agents.orchestrator.generate_embedding", new=embed))
        stack.enter_context(patch("app.agents.orchestrator.lookup_cache", new=AsyncMock(return_value=None)))
        stack.enter_context(patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)))
        stack.enter_context(patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)))
        stack.enter_context(patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)))

        runner = InMemoryRunner(app=App(name="verifact", root_agent=root_agent))
        await runner.session_service.create_session(app_name="verifact", user_id="u", session_id="s")
        message = types.Content(role="user", parts=[types.Part.from_text(text=json.dumps({"text": text}))])
        return [e async for e in runner.run_async(user_id="u", session_id="s", new_message=message)]


@pytest.mark.anyio
async def test_text_input_runs_the_pipeline_and_scores():
    calls: list = []
    embed = MagicMock(side_effect=lambda t: calls.append("embed") or [0.1] * 384)

    events = await _run(CLAIM, calls, embed)

    outcomes = [e.output for e in events if isinstance(e.output, PipelineOutcome)]
    assert len(outcomes) == 1 and outcomes[0].scoring_res is not None
    assert calls == ["warm", "embed"]          # models are warm before the request deadline starts


@pytest.mark.anyio
@pytest.mark.parametrize("text, error", [
    ("Is the Earth flat?", NonVerifiableClaimError),
    ("x" * 2001, InvalidInputError),
])
async def test_input_guardrails_apply_on_the_adk_web_path(text, error):
    embed = MagicMock()
    with pytest.raises(error):
        await _run(text, [], embed)
    embed.assert_not_called()
