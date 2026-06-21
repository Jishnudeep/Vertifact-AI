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
    Safely extract and parse JSON from LLM output, handling single quotes and surrounding markdown.
    """
    text = text.strip()
    # Find first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix keys with single quotes: 'key': -> "key":
        fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
        # Fix string values with single quotes: : 'value' -> : "value"
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        try:
            return json.loads(fixed)
        except Exception as exc:
            logger.error(f"parse_json_safely failed on text: {text}. Error: {exc}")
            raise

def extract_key_via_regex(text: str, key: str, is_bool: bool = False) -> Optional[Any]:
    """
    Extract a JSON property value using regex as a resilient fallback.
    """
    if is_bool:
        pattern = rf'["\']{key}["\']\s*:\s*(true|false)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
    else:
        # Match either double or single quoted values
        pattern = rf'["\']{key}["\']\s*:\s*["\']([^"\']*)["\']'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
            
        # Fallback to match even if the value has internal double quotes (loose match)
        pattern = rf'["\']{key}["\']\s*:\s*["\'](.*?)["\']\s*(?:,|\}})'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Match numeric values (from linguistic agent)
        pattern = rf'["\']{key}["\']\s*:\s*([0-9.]+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
            
    return None

def parse_adk_response(content: str, expected_keys: List[str], bool_keys: List[str] = []) -> Dict[str, Any]:
    """
    Parse the ADK response content. Tries json.loads first, then falls back to regex extraction
    to ensure maximum resilience against LLM formatting errors.
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
    Retrieve a cached ADK Runner, or construct and cache a new one if not present.
    """
    cache_key = f"{app_name}:{name}:{instruction}"
    if cache_key not in _runner_cache:
        logger.info(f"Creating new ADK Runner for agent: {name} (app: {app_name})")
        model = LiteLlm(model=settings.model_id, api_key=settings.groq_api_key)
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
    Uses cached runners and independent sessions per request.
    """
    runner = get_adk_runner(name, instruction, app_name)
    
    # Generate unique user_id and session_id per call to prevent conversation history cross-talk
    run_id = str(uuid.uuid4())[:8]
    user_id = f"{app_name}-user-{name}-{run_id}"
    session_id = f"{app_name}-session-{name}-{run_id}"
    
    await _session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""
            
    return final_text
