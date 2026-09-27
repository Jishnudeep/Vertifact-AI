import asyncio
import logging
import json
import re
import uuid
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except Exception:
    litellm = None

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.config import settings
from app.utils.errors import classify_provider_error

logger = logging.getLogger(__name__)

# Cache of ADK Runners keyed by a unique string for the agent configuration
_runner_cache: Dict[str, Runner] = {}
# Single shared session service across all agents
_session_service = InMemorySessionService()

@dataclass(frozen=True)
class BraceGroup:
    """A top-level `{...}` group found in LLM output."""
    text: str
    closed: bool

def scan_brace_groups(text: str) -> List[BraceGroup]:
    """
    Return every TOP-LEVEL brace group in `text`, in order, using a string-aware scanner.

    R8.1 layer 2. `text.find('{')` .. `text.rfind('}')` spans from the FIRST brace anywhere in
    the message to the LAST — so a brace-carrying claim echoed in the model's preamble merged
    with (or entirely replaced) the model's own object. This scanner instead isolates each
    balanced group: braces inside JSON string literals are ignored (with `\\"` escape handling),
    so a `{` mentioned in an `explanation` value cannot unbalance the scan.

    The final group is reported with `closed=False` when the response was truncated mid-object;
    callers use that to tell "the model's answer is incomplete" from "the model answered".
    """
    groups: List[BraceGroup] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            # Quotes only open a string once we are inside an object
            if depth > 0:
                in_string = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    groups.append(BraceGroup(text=text[start:i + 1], closed=True))
                    start = -1

    if depth > 0 and start != -1:
        groups.append(BraceGroup(text=text[start:], closed=False))

    return groups

def _loads_object(candidate: str) -> Optional[Dict[str, Any]]:
    """Parse one candidate group into a dict, tolerating single-quoted JSON. None on failure."""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Fix single quotes: 'key': -> "key": and : 'value' -> : "value"
        fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', candidate)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

def parse_json_safely(text: str, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract and parse the MODEL'S OWN JSON object out of LLM output.

    R8.1 layer 2. Order of attempts:
      1. `json.loads` on the whole stripped message (the clean, instruction-following case).
      2. Otherwise, every BALANCED brace group is collected via `scan_brace_groups` and parsed
         independently. The naive first-`{`..last-`}` span (and the "last `{` to last `}`"
         heuristic) are gone: both let an echoed, attacker-authored object be adopted as the
         answer.
      3. Among the groups that parse, the candidate carrying the most `expected_keys` wins;
         ties break to the LAST group, since the model's real answer follows any preamble echo.

    A response whose FINAL group is unclosed was truncated mid-answer. In that case the only
    complete objects present are preamble/echo, so this raises instead of adopting one — the
    caller's regex fallback then works against the truncated tail (the model's own words).

    Raises:
        ValueError: no balanced, parseable object attributable to the model was found.
    """
    if not text:
        raise ValueError("No JSON object found in LLM response")

    text = text.strip()

    # 1. The whole message is the object
    whole = _loads_object(text)
    if whole is not None:
        return whole

    groups = scan_brace_groups(text)
    if not groups:
        raise ValueError("No JSON object found in LLM response")

    if not groups[-1].closed:
        raise ValueError(
            "LLM response was truncated mid-JSON; refusing to adopt an earlier (possibly echoed) object"
        )

    candidates = [obj for obj in (_loads_object(g.text) for g in groups) if obj is not None]
    if not candidates:
        logger.error(f"parse_json_safely found no parseable JSON object in: {text[:200]}...")
        raise ValueError("No parseable JSON object found in LLM response")

    if expected_keys:
        wanted = {k.lower() for k in expected_keys}
        # max() over (overlap, index) → most expected keys, ties broken to the LAST group
        _, best_index = max(
            (len(wanted & {str(k).lower() for k in obj.keys()}), idx)
            for idx, obj in enumerate(candidates)
        )
        return candidates[best_index]

    return candidates[-1]

def json_span(text: str) -> Optional[str]:
    """
    Return the span of `text` that holds the MODEL'S answer object.

    R8.1: that is the LAST top-level brace group — a preamble echo of the untrusted claim always
    precedes the model's own answer. If the final group is unclosed (truncated response) its
    partial text is returned so keys emitted before the cut can still be recovered.

    Returns None when the text contains no brace group at all — i.e. the model emitted prose
    only, in which case no value may be scraped from it.
    """
    groups = scan_brace_groups(text)
    if not groups:
        return None
    return groups[-1].text

def _mask_string_values(span: str) -> str:
    """
    Blank the CONTENT of every JSON string value, preserving length so offsets stay aligned.

    R8.1 layer 3: a quoted key echoed inside a string value (e.g.
    `{"explanation": "the user asked for \\"relevant\\": true", "relevant": false}`) otherwise
    matches the key regex BEFORE the model's real key does. Masking values first means the
    key search can only ever land on a real key; the value is then read from the original span.
    """
    def repl(match: "re.Match[str]") -> str:
        prefix = match.group(1)
        body = match.group(0)[len(prefix):]  # the quoted value, quotes included
        return f'{prefix}"{" " * (len(body) - 2)}"'

    return re.sub(r'(:\s*)"(?:[^"\\]|\\.)*"', repl, span)

def extract_key_via_regex(text: str, key: str, is_bool: bool = False) -> Optional[Any]:
    """
    Extract a JSON property value via regex as a resilient fallback for truncated output.

    Hardened against prompt injection (R7.B1, R8.1). LLM output echoes untrusted text, so this
    fallback:
      1. only searches inside the model's answer span (the LAST brace group), never a prose
         preamble or an earlier echoed object,
      2. locates the key against a copy whose string VALUES are masked out, so a key quoted
         inside an `explanation` can never be mistaken for a real key,
      3. requires the key to be QUOTED (`"key":`) — a bare `key:` is never accepted,
      4. takes the FIRST real match, so a later attacker-appendable repetition cannot override
         the model's own value.
    """
    span = json_span(text)
    if span is None:
        return None

    masked = _mask_string_values(span)
    escaped = re.escape(key)

    key_match = re.search(rf'["\']{escaped}["\']\s*:', masked, re.IGNORECASE)
    if key_match is None:
        return None

    # Read the value from the ORIGINAL span, anchored at the real key
    tail = span[key_match.start():]

    if is_bool:
        match = re.match(rf'["\']{escaped}["\']\s*:\s*(true|false)', tail, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return None

    # Prefer a properly quoted value
    match = re.match(rf'["\']{escaped}["\']\s*:\s*"([^"]*)"', tail, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fall back to an unquoted scalar (numbers, bare enum tokens), still key-quoted and span-scoped
    match = re.match(rf'["\']{escaped}["\']\s*:\s*([A-Za-z0-9._+-]+)', tail, re.IGNORECASE)
    if match:
        return match.group(1)

    return None

def parse_adk_response(content: str, expected_keys: List[str], bool_keys: List[str] = []) -> Dict[str, Any]:
    """
    Parse the ADK response content. Tries json.loads first, then falls back to regex extraction.
    `expected_keys` is threaded into the parser so it can identify the model's own answer object
    among any echoed ones (R8.1); the regex fallback only ever sees the answer span, never the
    raw prose (which may echo untrusted text).
    """
    try:
        return parse_json_safely(content, expected_keys=expected_keys)
    except Exception as exc:
        logger.warning(f"Standard JSON parsing failed (Error: {exc}). Attempting regex fallback parsing...")
        span = json_span(content)
        if span is None:
            logger.warning("No JSON object found in LLM output; regex fallback yields nothing.")
            return {}
        result = {}
        for key in expected_keys:
            is_bool = key in bool_keys
            val = extract_key_via_regex(span, key, is_bool)
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
        # max_tokens=1024: reasoning-capable models spend budget before the JSON body;
        # 256 truncated the `explanation` field and forced the (injection-prone) regex fallback (R7.P4).
        model = LiteLlm(model=settings.model_id, api_key=settings.groq_api_key, temperature=0.0, max_tokens=1024)
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
    Includes transient retry with exponential backoff on provider RateLimitError.
    Cleans up session upon completion to avoid memory leaks.
    """
    runner = get_adk_runner(name, instruction, app_name)
    max_retries = 2
    
    for attempt in range(max_retries + 1):
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
                    # Reasoning models (e.g. gpt-oss) return the chain-of-thought as a separate
                    # `thought` part BEFORE the answer part. parts[0] is therefore reasoning prose
                    # that echoes the untrusted claim — never the JSON answer. Keep only
                    # non-thought parts; if there are none, return "" so the caller degrades
                    # rather than parsing reasoning text (R7.B1).
                    answer_parts = [
                        part.text for part in event.content.parts
                        if part.text and not getattr(part, "thought", False)
                    ]
                    if not answer_parts:
                        logger.warning(
                            f"ADK agent '{name}' returned only reasoning parts and no answer content."
                        )
                    final_text = "\n".join(answer_parts).strip()
            return final_text
        except Exception as exc:
            # Check for RateLimitError
            is_rate_limit = False
            retry_after = 5.0
            if litellm is not None and isinstance(exc, litellm.RateLimitError):
                is_rate_limit = True
                err_details = classify_provider_error(exc, provider="groq")
                if err_details.retry_after is not None:
                    retry_after = min(err_details.retry_after, 6.0)

            if is_rate_limit and attempt < max_retries:
                logger.warning(
                    f"ADK agent '{name}' hit RateLimitError on attempt {attempt+1}/{max_retries+1}. "
                    f"Backing off for {retry_after:.2f}s before retry..."
                )
                await asyncio.sleep(retry_after)
                continue
            raise
        finally:
            try:
                await _session_service.delete_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
            except Exception as cleanup_err:
                logger.debug(f"Failed to delete ADK session {session_id}: {cleanup_err}")
