"""prepare_verify_request: the shared front half of verify_claim and the adk web entry point."""
import time
from unittest.mock import patch

import pytest

from app.agents.orchestrator import (
    InvalidInputError,
    NonVerifiableClaimError,
    VerifyRequest,
    prepare_verify_request,
)
from app.config import settings
from app.utils.validators import MAX_INPUT_LENGTH


@pytest.mark.anyio
async def test_returns_sanitized_request_with_embedding():
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384) as embed:
        req = await prepare_verify_request("<script>alert(1)</script>The Earth orbits the Sun", bypass_cache=True)

    assert isinstance(req, VerifyRequest)
    assert req.claim == "The Earth orbits the Sun"
    assert req.embedding == [0.1] * 384
    assert req.bypass_cache is True
    embed.assert_called_once_with("The Earth orbits the Sun")


@pytest.mark.anyio
async def test_deadline_is_set_before_the_embedding_runs():
    """The embedding is part of the request budget, exactly as in the old inline code."""
    seen = []

    def fake_embedding(text):
        seen.append(time.monotonic())
        # Windows' monotonic clock ticks every ~16 ms; without a visible gap a deadline taken
        # *after* the embedding would land on the same tick and this test could not catch it.
        time.sleep(0.05)
        return [0.1] * 384

    with patch("app.agents.orchestrator.generate_embedding", side_effect=fake_embedding):
        req = await prepare_verify_request("The Earth orbits the Sun")

    assert req.deadline <= seen[0] + settings.request_deadline_s


@pytest.mark.anyio
@pytest.mark.parametrize("text, error", [
    ("   ", InvalidInputError),
    ("x" * (MAX_INPUT_LENGTH + 1), InvalidInputError),
    ("Is the Earth flat?", NonVerifiableClaimError),
    ("I think coffee is the greatest drink in human history.", NonVerifiableClaimError),
])
async def test_rejected_input_raises_before_any_embedding(text, error):
    with patch("app.agents.orchestrator.generate_embedding") as embed:
        with pytest.raises(error):
            await prepare_verify_request(text)
    embed.assert_not_called()
