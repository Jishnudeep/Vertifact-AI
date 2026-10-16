"""ADK web entry point for the VeriFact verification pipeline.

`adk web` run from `backend/` discovers this package via `root_agent`. It takes the
same `{"text": ...}` input as the API's claim text, prepares the request (sanitize,
validate, embed), then runs the pipeline in `app.agents.orchestrator`.
"""

import asyncio

from google.adk.workflow import Workflow, node
from pydantic import BaseModel

from app.agents.orchestrator import VerifyRequest, build_verification_workflow, prepare_verify_request
from app.services.warmup import ensure_models_warm


class ClaimInput(BaseModel):
    text: str


@node(name="prepare_request")
async def prepare_request_node(node_input: ClaimInput) -> VerifyRequest:
    # adk web has no startup hook (main.py's lifespan does not run). Load the models here, before
    # prepare_verify_request starts the request deadline, so a cold load is never charged to it.
    await asyncio.to_thread(ensure_models_warm)
    return await prepare_verify_request(node_input.text)


root_agent = Workflow(
    name="verifact",
    description="Claim text -> prepare request -> verification pipeline.",
    input_schema=ClaimInput,
    edges=[
        ("START", prepare_request_node),
        (prepare_request_node, build_verification_workflow()),
    ],
)
