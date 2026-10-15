import threading
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Callable, Optional

from app.config import settings


class LLMBudgetExceeded(RuntimeError):
    """The daily LLM call budget is spent. Agents degrade; they never guess."""


class DailyBudget:
    """
    A per-UTC-day counter. try_charge() is atomic and charges nothing when it refuses.
    In-process only: it resets on restart and is not shared across instances, which is
    acceptable for the single-instance free-tier deployment (TDD §11).
    """

    def __init__(self, name: str, limit: int, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.name = name
        self.limit = limit
        self._clock = clock
        self._day: Optional[date] = None
        self._used = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = self._clock().date()
        if today != self._day:
            self._day, self._used = today, 0

    def try_charge(self, units: int = 1) -> bool:
        with self._lock:
            self._roll()
            if self.limit > 0 and self._used + units > self.limit:
                return False
            self._used += units
            return True

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used



@lru_cache()
def get_llm_budget() -> DailyBudget:
    return DailyBudget("llm_calls", settings.llm_daily_call_budget)


@lru_cache()
def get_search_budget() -> DailyBudget:
    """Gemini grounded-search calls per UTC day (Plan 6). Exhaustion is an evidence outage."""
    return DailyBudget("search_calls", settings.search_daily_call_budget)
