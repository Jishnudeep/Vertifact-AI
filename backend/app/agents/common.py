"""Small shared pieces for the worker agents: the model seam, the budget callback, prompt fencing, log helpers."""
import logging
from typing import Any, Dict, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from app.config import settings
from app.services import budget
from app.utils.validators import neutralize_for_prompt

logger = logging.getLogger(__name__)

# Angle brackets in untrusted text become look-alikes so a fetched page cannot close its own fence;
# double quotes too, for text placed inside an attribute value.
_FENCE_SAFE = str.maketrans({"<": "‹", ">": "›"})
_ATTR_SAFE = str.maketrans({"<": "‹", ">": "›", '"': "'"})


def fenced(text: str) -> str:
    """Untrusted text for the body of a fence: JSON-answer syntax and angle brackets neutralised."""
    return neutralize_for_prompt(text).translate(_FENCE_SAFE)


def fenced_attr(text: str) -> str:
    return neutralize_for_prompt(text).translate(_ATTR_SAFE)


def get_model() -> Any:
    """
    The model of every worker LlmAgent: a Gemini model id (ADK's native Gemini, which retries
    transient errors itself). This is the one seam tests and the eval replay patch.
    """
    return settings.model_id


def charge_llm_budget(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """before_model_callback: one daily-budget charge per model turn; an exhausted budget raises."""
    if not budget.get_llm_budget().try_charge():
        raise budget.LLMBudgetExceeded("Daily LLM call budget exhausted")
    return None


def clip(text: Optional[str], limit: int = 80) -> str:
    """Untrusted text (a claim, a query) for a log line: one line, at most `limit` chars."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log_model_turn(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """before_model_callback: one INFO line per model turn (turn = prior model messages + 1)."""
    contents = getattr(llm_request, "contents", None) or []
    turn = 1 + sum(1 for c in contents if getattr(c, "role", None) == "model")
    logger.info("%s: model turn %d", getattr(callback_context, "agent_name", "?"), turn)
    return None


def log_model_response(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """after_model_callback: what the turn produced (tool calls / final answer / error). Never the text."""
    if getattr(llm_response, "partial", None):
        return None
    agent = getattr(callback_context, "agent_name", "?")
    if getattr(llm_response, "error_code", None):
        logger.warning("%s: model returned error %s", agent, llm_response.error_code)
        return None
    content = getattr(llm_response, "content", None)
    parts = (content.parts if content is not None else None) or []
    calls = [p.function_call.name for p in parts if getattr(p, "function_call", None) is not None]
    if calls:
        logger.info("%s: model requested tool(s): %s", agent, ", ".join(str(c) for c in calls))
    else:
        logger.info("%s: model returned its final answer", agent)
    return None


def agent_kwargs() -> Dict[str, Any]:
    """Callbacks every worker LlmAgent is built with (the latency bench wraps this to time tools)."""
    return {
        "before_model_callback": [log_model_turn, charge_llm_budget],
        "after_model_callback": log_model_response,
    }
