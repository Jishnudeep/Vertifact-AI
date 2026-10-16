from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import main
from app.services import warmup


def _patched_startup(spacy_loader):
    """Stub every heavy startup step except the one under test."""
    stack = ExitStack()
    stack.enter_context(patch.object(warmup, "get_spacy_nlp", spacy_loader))
    stack.enter_context(patch.object(warmup, "load_bias_ratings"))
    stack.enter_context(patch.object(warmup, "generate_embedding"))
    stack.enter_context(patch.object(warmup, "get_nli_model"))
    runner = stack.enter_context(patch.object(main, "get_verification_runner"))
    stack.enter_context(patch.object(main, "drain_background_tasks", AsyncMock()))
    return stack, runner


@pytest.mark.anyio
async def test_lifespan_preloads_spacy():
    """The agents rely on startup having loaded spaCy; a cold load inside a request costs seconds."""
    spacy_loader = MagicMock()
    stack, _ = _patched_startup(spacy_loader)
    with stack:
        async with main.lifespan(main.app):
            spacy_loader.assert_called_once_with()


@pytest.mark.anyio
async def test_lifespan_fails_when_spacy_preload_fails():
    """
    en_core_web_sm is a pinned dependency, so a failed load is a broken install, not a transient
    error. Booting anyway would degrade Cross-Ref and Linguistic on every request; fail at boot.
    """
    stack, runner = _patched_startup(MagicMock(side_effect=RuntimeError("model not installed")))
    with stack, pytest.raises(RuntimeError, match="model not installed"):
        async with main.lifespan(main.app):
            pass
    runner.assert_not_called()
