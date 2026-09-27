from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from app.config import settings

# Every worker agent whose outputs orchestrator.verify_claim persists in results.agent_outputs.
_AGENTS = ("factcheck", "crossref", "linguistic")


def ttl_for(evidence_status: str) -> timedelta:
    """Fact-checked results live longer; news-driven statuses expire within a day."""
    if evidence_status == "fact_checked":
        return timedelta(hours=settings.cache_ttl_fact_checked_hours)
    return timedelta(hours=settings.cache_ttl_default_hours)


def is_complete_run(agent_outputs: Any) -> bool:
    """True only when every agent's persisted output says degraded=False. Missing data fails closed."""
    if not isinstance(agent_outputs, dict):
        return False
    return all(
        isinstance(agent_outputs.get(agent), dict) and agent_outputs[agent].get("degraded") is False
        for agent in _AGENTS
    )


def is_fresh(evidence_status: str, created_at: Union[str, datetime, None], now: datetime) -> bool:
    """True when the cached result is younger than its TTL. Unparseable timestamps fail closed."""
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return False
    if not isinstance(created_at, datetime):
        return False
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return now - created_at <= ttl_for(evidence_status)


def unservable_reason(row: Dict[str, Any], now: datetime) -> Optional[str]:
    """
    Cheap, pure checks run before the (costlier) equivalence check.
    Returns None when the row may be served, otherwise a short reason for the log.
    """
    conf = str(row.get("confidence_level", "")).lower()
    if conf in {"low", "insufficient"}:
        return f"confidence is '{conf}'"
    status = row.get("evidence_status")
    if not status:
        return "predates stance-aware scoring (no evidence_status)"
    if not is_complete_run(row.get("agent_outputs")):
        return "partial analysis (an agent degraded)"
    if not is_fresh(status, row.get("created_at"), now):
        return "stale (past TTL)"
    return None
