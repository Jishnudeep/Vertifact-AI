from unittest.mock import MagicMock, patch

import pytest

from app.db.purge import purge_old_claims


@pytest.mark.anyio
async def test_purge_calls_rpc_with_configured_days():
    supabase = MagicMock()
    supabase.rpc.return_value.execute.return_value = MagicMock(data=7)
    with patch("app.db.purge.get_supabase", return_value=supabase):
        removed = await purge_old_claims()
    supabase.rpc.assert_called_once_with("purge_old_claims", {"retention_days": 90})
    assert removed == 7


@pytest.mark.anyio
async def test_purge_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        await purge_old_claims(0)
