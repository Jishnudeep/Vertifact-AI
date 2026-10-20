"""
Test doubles for the worker agents.

`ScriptedLlm` plays scripted model turns (a tool call, a final JSON answer, or an exception) and is
installed through the single model seam `app.agents.common.get_model`. `run_node` runs one worker node
through a real ADK workflow + supervisor, like the orchestrator does.

Pass `*steps` for one agent, or `routes={instruction substring: [steps]}` when several agents run
(each gets its own cursor).
"""
import json
from typing import Any, Dict, List, Optional

from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node
from google.genai import types
from pydantic import Field, PrivateAttr

from app.agents.request import VerifyRequest


def call(name: str, **args: Any) -> List[types.Part]:
    """One model turn that calls the tool `name`."""
    return [types.Part(function_call=types.FunctionCall(name=name, args=args))]


def final(**fields: Any) -> List[types.Part]:
    """The model's final structured answer: JSON text matching the agent's output_schema (Gemini-native)."""
    return [types.Part(text=json.dumps(fields))]


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    steps: List[Any] = Field(default_factory=list)
    routes: Dict[str, List[Any]] = Field(default_factory=dict)
    seen: List[Any] = Field(default_factory=list)          # every LlmRequest received
    _cursors: Dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)    # like Gemini: the final answer is JSON text

    @property
    def turns(self) -> int:
        return len(self.seen)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        self.seen.append(llm_request)
        key, steps = "", self.steps
        if self.routes:
            instruction = str(getattr(llm_request.config, "system_instruction", "") or "")
            key = next((k for k in self.routes if k in instruction), None)
            if key is None:
                raise AssertionError(f"ScriptedLlm has no route for instruction: {instruction[:120]!r}")
            steps = self.routes[key]
        index = self._cursors.get(key, 0)
        self._cursors[key] = index + 1
        if index >= len(steps):
            raise AssertionError(f"ScriptedLlm has no step for model turn {index + 1} (route {key!r})")
        step = steps[index]
        if isinstance(step, Exception):
            raise step
        yield LlmResponse(content=types.Content(role="model", parts=step))


def install_script(monkeypatch, *steps: Any, routes: Optional[Dict[str, List[Any]]] = None) -> ScriptedLlm:
    llm = ScriptedLlm(steps=list(steps), routes=dict(routes or {}))
    monkeypatch.setattr("app.agents.common.get_model", lambda: llm)
    return llm


async def run_node(worker_node, claim: str):
    """Run a worker node as a dynamic child of a supervisor. Returns its result or raises its (unwrapped) error."""
    seen: Dict[str, Any] = {}

    @node(name="supervisor", rerun_on_resume=True)
    async def supervisor(ctx: Context, node_input: VerifyRequest) -> str:
        try:
            seen["result"] = await ctx.run_node(worker_node, node_input)
        except Exception as exc:  # noqa: BLE001
            seen["error"] = getattr(exc, "error", None) if isinstance(getattr(exc, "error", None), Exception) else exc
        return "done"

    workflow = Workflow(name="test_wf", input_schema=VerifyRequest, edges=[("START", supervisor)])
    runner = InMemoryRunner(app=App(name="test_app", root_agent=workflow))
    await runner.session_service.create_session(app_name="test_app", user_id="u", session_id="s")
    request = VerifyRequest(claim=claim, embedding=[0.1] * 4, deadline=0.0)
    message = types.Content(role="user", parts=[types.Part.from_text(text=request.model_dump_json())])
    async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
        pass
    if "error" in seen:
        raise seen["error"]
    return seen["result"]
