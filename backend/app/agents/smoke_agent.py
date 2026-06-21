"""Minimal ADK + LiteLLM + Groq smoke test.

A throwaway "does the wiring work?" agent — NOT a VeriFact verification agent.
It exists to confirm three things end-to-end before we build the real pipeline:

  1. google-adk can construct an LlmAgent.
  2. LiteLlm routes that agent to Groq (Llama 3.1 8B) using the key from config.
  3. The ADK Runner can drive a turn and return a final response.

Run it:
    cd backend
    uv run python -m app.agents.smoke_agent

Note on the model: Groq does not serve a 3B Llama. The project default
(`settings.model_id`) is `groq/llama-3.1-8b-instant`, and per the project rule the
model id always comes from config — never hardcoded in agent logic.
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.config import settings

APP_NAME = "verifact-smoke"

# LiteLLM reads the provider key from the process environment. pydantic-settings
# loads it into `settings` but not into os.environ, so export it here. We also pass
# api_key explicitly to LiteLlm as a belt-and-suspenders measure.
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)


def build_agent() -> LlmAgent:
    """Construct the smoke-test agent. Model id comes from config (LiteLLM-routed)."""
    return LlmAgent(
        name="verifact_smoke",
        model=LiteLlm(model=settings.model_id, api_key=settings.groq_api_key),
        description="Throwaway agent to verify ADK + LiteLLM + Groq wiring.",
        instruction=(
            "You are a concise assistant. Answer the user's question directly "
            "in a single short sentence. Do not elaborate."
        ),
    )


async def run_once(prompt: str) -> str:
    """Drive one turn through the ADK Runner and return the final text response."""
    agent = build_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_id, session_id = "smoke-user", "smoke-session"
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

    message = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""

    return final_text


async def _main() -> None:
    prompt = "What is 17 + 25? Reply with just the number and a five-word fun fact."
    print(f"Model:  {settings.model_id}")
    print(f"Prompt: {prompt}\n")
    try:
        answer = await run_once(prompt)
    except Exception as exc:  # smoke test: surface the failure clearly
        print(f"FAILED: {type(exc).__name__}: {exc}")
        raise

    print(f"Response: {answer}")
    print("\nADK + LiteLLM + Groq wiring OK." if answer else "\nNo response text returned.")


if __name__ == "__main__":
    asyncio.run(_main())
