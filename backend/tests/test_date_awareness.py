"""Tests for date awareness (Task A.2)."""
import re
from unittest.mock import AsyncMock

import pytest
from google.adk.agents.context import Context

from app.agents import common
from app.agents import crossref_agent as cr
from app.agents import factcheck_agent as fc


def test_get_today_utc_format():
    today = common.get_today_utc()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", today)


def test_factcheck_instruction_contains_injected_date(monkeypatch):
    monkeypatch.setattr(common, "get_today_utc", lambda: "2026-10-20")
    instruction = fc.get_instruction()
    assert "Today's date is 2026-10-20 (UTC)" in instruction
    assert "fact-check lookup agent" in instruction


def test_crossref_instruction_contains_injected_date(monkeypatch):
    monkeypatch.setattr(common, "get_today_utc", lambda: "2026-10-20")
    instruction = cr.get_instruction()
    assert "Today's date is 2026-10-20 (UTC)" in instruction
    assert "cross-reference agent" in instruction


@pytest.mark.anyio
async def test_factcheck_agent_uses_patched_clock(monkeypatch):
    """The agent instruction must reflect the app-supplied date; unpatched clock must not be used."""
    monkeypatch.setattr(common, "get_today_utc", lambda: "2026-05-15")

    captured_instruction = None

    async def fake_run_node(agent, input_text, *args, **kwargs):
        nonlocal captured_instruction
        captured_instruction = agent.instruction
        return fc.FactCheckDecision(
            evidence="none",
            relevant=False,
            rating="unproven",
            explanation="test",
        )

    ctx = AsyncMock(spec=Context)
    ctx.run_node = AsyncMock(side_effect=fake_run_node)
    monkeypatch.setattr(fc, "finalize", lambda claim, decision, ev: fc.make_neutral_factcheck())

    await fc.analyze(ctx, "Test claim")
    assert captured_instruction is not None
    assert "Today's date is 2026-05-15 (UTC)" in captured_instruction


@pytest.mark.anyio
async def test_crossref_agent_uses_patched_clock(monkeypatch):
    """The agent instruction must reflect the app-supplied date; unpatched clock must not be used."""
    monkeypatch.setattr(common, "get_today_utc", lambda: "2026-05-15")
    monkeypatch.setattr(cr, "_extract_entities", lambda claim: [])

    captured_instruction = None

    async def fake_run_node(agent, input_text, *args, **kwargs):
        nonlocal captured_instruction
        captured_instruction = agent.instruction
        return cr.CoverageSummary(queries_rationale="test")

    ctx = AsyncMock(spec=Context)
    ctx.run_node = AsyncMock(side_effect=fake_run_node)

    # Prevent finalize from failing on empty runs
    monkeypatch.setattr(cr, "finalize", lambda entities, cov: cr.make_neutral_crossref())

    await cr.analyze(ctx, "Test claim")
    assert captured_instruction is not None
    assert "Today's date is 2026-05-15 (UTC)" in captured_instruction
