"""
Patch the three workers' bodies for orchestrator-level tests. Each worker module has an
`analyze(ctx, claim)`; the real node (timeout, dynamic child run) still executes around the fake.
Each fake is `async fn(claim) -> result` (an AsyncMock works).
"""
from contextlib import ExitStack, contextmanager
from typing import Callable, Optional
from unittest.mock import patch


@contextmanager
def patched_agents(fc: Optional[Callable] = None, cr: Optional[Callable] = None, ling: Optional[Callable] = None):
    with ExitStack() as stack:
        for module, impl in (("factcheck_agent", fc), ("crossref_agent", cr), ("linguistic_agent", ling)):
            if impl is not None:
                async def fake(ctx, claim, _impl=impl):
                    return await _impl(claim)
                stack.enter_context(patch(f"app.agents.{module}.analyze", new=fake))
        yield
