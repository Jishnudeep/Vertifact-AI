import logging
import json
import re
import uuid
from typing import Optional, List, Dict, Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# Cache of ADK Runners keyed by a unique string for the agent configuration
_runner_cache: Dict[str, Runner] = {}
# Single shared session service across all agents
_session_service = InMemorySessionService()

def parse_json_safely(text: str) -> Dict[str, Any]:
    """
    Safely extract and parse JSON from LLM output, handling markdown blocks, single quotes, and preamble.
    """
    text = text.strip()
    
    # Strip markdown codeblocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            
    # Try parsing from first '{' to last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_candidate = text[start:end+1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass
            
    # Sometime preamble text has a '{', try parsing from the LAST '{' to last '}'
    last_start = text.rfind('{')
    if last_start != -1 and end != -1 and last_start < end:
        try:
            return json.loads(text[last_start:end+1])
        except json.JSONDecodeError:
            pass
            
    # Fix single quotes: 'key': -> "key": and : 'value' -> : "value"
    if start != -1 and end != -1:
        fixed = text[start:end+1]
        fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        try:
            return json.loads(fixed)
        except Exception as exc:
            logger.error(f"parse_json_safely failed on text: {text[:200]}... Error: {exc}")
            raise
            
    raise ValueError("No JSON object found in LLM response")

def extract_key_via_regex(text: str, key: str, is_bool: bool = False) -> Optional[Any]:
    """
    Extract a JSON property value using regex as a resilient fallback, preferring the last match.
    """
    if is_bool:
        pattern = rf'["\']?{key}["\']?\s*:\s*(true|false)'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).lower() == "true"
    else:
        # Match "key": "value" or key: "value" or key: value
        pattern = rf'["\']?{key}["\']?\s*:\s*["\']([^"\']+)["\']'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
            
        pattern = rf'["\']?{key}["\']?\s*:\s*([a-zA-Z0-9._]+)'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
            
    return None

def parse_adk_response(content: str, expected_keys: List[str], bool_keys: List[str] = []) -> Dict[str, Any]:
    """
    Parse the ADK response content. Tries json.loads first, then falls back to regex extraction.
    """
    try:
        return parse_json_safely(content)
    except Exception as exc:
        logger.warning(f"Standard JSON parsing failed (Error: {exc}). Attempting regex fallback parsing...")
        result = {}
        for key in expected_keys:
            is_bool = key in bool_keys
            val = extract_key_via_regex(content, key, is_bool)
            if val is not None:
                result[key] = val
        return result

def get_adk_runner(name: str, instruction: str, app_name: str) -> Runner:
    """
    Retrieve a cached ADK Runner with temperature=0.0 for deterministic evaluation.
    """
    cache_key = f"{app_name}:{name}:{instruction}"
    if cache_key not in _runner_cache:
        logger.info(f"Creating new ADK Runner for agent: {name} (app: {app_name}) with temperature=0.0")
        model = LiteLlm(model=settings.model_id, api_key=settings.groq_api_key, temperature=0.0, max_tokens=256)
        agent = LlmAgent(
            name=name,
            model=model,
            description=f"ADK Agent for {name}",
            instruction=instruction,
        )
        runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=_session_service,
        )
        _runner_cache[cache_key] = runner
    else:
        logger.debug(f"Reusing cached ADK Runner for agent: {name} (app: {app_name})")
    return _runner_cache[cache_key]

async def run_adk_agent(name: str, instruction: str, prompt: str, app_name: str) -> str:
    """
    Helper to run a Google ADK LlmAgent and retrieve the final text response.
    Cleans up session upon completion to avoid memory leaks.
    """
    runner = get_adk_runner(name, instruction, app_name)
    
    run_id = str(uuid.uuid4())[:8]
    user_id = f"{app_name}-user-{name}-{run_id}"
    session_id = f"{app_name}-session-{name}-{run_id}"
    
    await _session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    try:
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""
        return final_text
    finally:
        try:
            await _session_service.delete_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except Exception as cleanup_err:
            logger.debug(f"Failed to delete ADK session {session_id}: {cleanup_err}")
