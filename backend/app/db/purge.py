"""
Retention purge (H4). Run manually or from a scheduler:
    uv run python -m app.db.purge
Prefer the pg_cron schedule in migration 05 when available.
"""
import asyncio
import logging
from typing import Optional

from app.config import settings
from app.db.connection import get_supabase

logger = logging.getLogger(__name__)


async def purge_old_claims(retention_days: Optional[int] = None) -> int:
    days = settings.retention_days if retention_days is None else retention_days
    if days < 1:
        raise ValueError("retention_days must be >= 1")
    supabase = get_supabase()
    response = await asyncio.to_thread(
        supabase.rpc("purge_old_claims", {"retention_days": days}).execute
    )
    removed = int(response.data or 0)
    logger.info(f"Retention purge removed {removed} claims older than {days} days.")
    return removed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(purge_old_claims()))
