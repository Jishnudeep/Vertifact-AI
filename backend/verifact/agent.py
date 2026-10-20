"""ADK web entry point for the VeriFact verification pipeline.

`adk web` run from `backend/` discovers this package via `root_agent`. It takes the
same `{"text": ...}` input as the API's claim text, prepares the request (sanitize,
validate, embed), then runs the pipeline in `app.agents.orchestrator`.
"""

import asyncio
import logging

# `adk web` normally configures the root logger itself (at its --log_level); this only covers an
# entry that did not, so the pipeline's INFO trace is visible. Idempotent: no-op once handlers exist.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from google.adk.workflow import Workflow, node  # noqa: E402  (after logging setup)
from pydantic import BaseModel  # noqa: E402

from app.agents.common import clip  # noqa: E402
from app.agents.orchestrator import VerifyRequest, build_verification_workflow, prepare_verify_request  # noqa: E402
from app.services.warmup import ensure_models_warm  # noqa: E402

logger = logging.getLogger(__name__)


class ClaimInput(BaseModel):
    text: str


@node(name="prepare_request")
async def prepare_request_node(node_input: ClaimInput) -> VerifyRequest:
    # adk web has no startup hook (main.py's lifespan does not run). Load the models here, before
    # prepare_verify_request starts the request deadline, so a cold load is never charged to it.
    await asyncio.to_thread(ensure_models_warm)
    request = await prepare_verify_request(node_input.text)
    logger.info("prepare_request: claim=%r (%d chars) ready", clip(request.claim), len(request.claim))
    return request


root_agent = Workflow(
    name="verifact",
    description="Claim text -> prepare request -> verification pipeline.",
    input_schema=ClaimInput,
    edges=[
        ("START", prepare_request_node),
        (prepare_request_node, build_verification_workflow()),
    ],
)
