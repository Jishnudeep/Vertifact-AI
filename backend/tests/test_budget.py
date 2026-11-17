from datetime import datetime, timedelta, timezone

import pytest

from app.services import budget as budget_mod
from app.services.budget import DailyBudget, LLMBudgetExceeded


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_budget_charges_until_limit_then_refuses():
    b = DailyBudget("t", limit=3, clock=Clock())
    assert b.try_charge(2) and b.used == 2
    assert not b.try_charge(2)          # would exceed; nothing charged
    assert b.used == 2
    assert b.try_charge(1) and b.used == 3


def test_budget_resets_on_new_utc_day():
    clock = Clock()
    b = DailyBudget("t", limit=1, clock=clock)
    assert b.try_charge()
    assert not b.try_charge()
    clock.now += timedelta(days=1)
    assert b.try_charge()


def test_zero_limit_disables_guard():
    b = DailyBudget("t", limit=0, clock=Clock())
    assert all(b.try_charge(100) for _ in range(5))


def test_charge_llm_budget_callback_charges_per_turn_and_refuses_when_exhausted(monkeypatch):
    from app.agents.common import charge_llm_budget
    b = DailyBudget("llm", limit=2, clock=Clock())
    monkeypatch.setattr(budget_mod, "get_llm_budget", lambda: b)
    assert charge_llm_budget(None, None) is None and charge_llm_budget(None, None) is None
    with pytest.raises(LLMBudgetExceeded):
        charge_llm_budget(None, None)


def test_turn_cap_refuses_turn_limit_plus_one_and_counts_per_instance():
    from app.agents.common import cap_model_turns
    from app.services.budget import ModelTurnLimitExceeded
    first, second = cap_model_turns(2), cap_model_turns(2)
    assert first(None, None) is None and first(None, None) is None
    with pytest.raises(ModelTurnLimitExceeded, match="2 turns"):
        first(None, None)
    assert second(None, None) is None  # a separate run keeps its own count


def test_agent_kwargs_builds_a_fresh_turn_cap_ahead_of_the_budget_charge(monkeypatch):
    from app.agents import common
    from app.services.budget import ModelTurnLimitExceeded
    b = DailyBudget("llm", limit=100, clock=Clock())
    monkeypatch.setattr(budget_mod, "get_llm_budget", lambda: b)
    monkeypatch.setattr(common.settings, "max_model_turns_per_agent", 1)
    callbacks = common.agent_kwargs()["before_model_callback"]
    assert callbacks.index(common.charge_llm_budget) > 1  # log, cap, then charge
    cap = callbacks[1]
    assert cap(None, None) is None
    with pytest.raises(ModelTurnLimitExceeded):
        cap(None, None)
    assert common.agent_kwargs()["before_model_callback"][1] is not cap


def test_turn_limit_is_classified_as_a_model_failure_not_quota():
    from app.services.budget import ModelTurnLimitExceeded
    from app.utils.errors import ProviderErrorCode, classify_provider_error
    details = classify_provider_error(ModelTurnLimitExceeded("Model did not finish within 6 turns"), "gemini")
    assert details.code == ProviderErrorCode.PARSE_ERROR and "6 turns" in details.message
