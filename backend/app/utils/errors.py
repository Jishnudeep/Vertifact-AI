import asyncio
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any, Dict

import httpx

try:
    import litellm
except ImportError:
    litellm = None

try:
    from google.genai import errors as genai_errors
except ImportError:
    genai_errors = None

from app.services.budget import LLMBudgetExceeded
from app.services.errors import EvidenceBudgetExceeded

class ProviderErrorCode(str, Enum):
    RATE_LIMIT = "RATE_LIMIT"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_ERROR = "AUTH_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN = "UNKNOWN"

@dataclass
class AgentErrorDetails:
    code: ProviderErrorCode
    provider: str
    message: str
    retry_after: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["code"] = self.code.value
        return d

_DEGRADATION_MESSAGES: Dict[str, str] = {
    ProviderErrorCode.RATE_LIMIT.value: "{agent_label} check rate limited",
    ProviderErrorCode.QUOTA_EXHAUSTED.value: "{agent_label} check quota exhausted",
    ProviderErrorCode.AUTH_ERROR.value: "{agent_label} authentication failed",
    ProviderErrorCode.MODEL_NOT_FOUND.value: "{agent_label} model unavailable",
    ProviderErrorCode.TIMEOUT.value: "{agent_label} check timed out",
    ProviderErrorCode.SERVICE_UNAVAILABLE.value: "{agent_label} service unavailable",
    ProviderErrorCode.PARSE_ERROR.value: "{agent_label} response could not be parsed",
    ProviderErrorCode.UNKNOWN.value: "{agent_label} check failed",
}

def degradation_message(agent_label: str, error_code: Optional[str]) -> str:
    """
    Returns a fixed sanitized degradation message per error code.
    UNKNOWN or None returns f"{agent_label} check failed".
    Never includes raw exception text.
    """
    if not error_code:
        return f"{agent_label} check failed"
    template = _DEGRADATION_MESSAGES.get(error_code)
    if template:
        return template.format(agent_label=agent_label)
    return f"{agent_label} check failed"

def _display_name(provider: str) -> str:
    """'gemini_search' -> 'Gemini search', 'google_factcheck' -> 'Google factcheck'."""
    return provider.replace("_", " ").strip().capitalize() or "Provider"

def _classify_genai_api_error(exc: BaseException, provider: str) -> Optional[AgentErrorDetails]:
    """Map a google-genai APIError (or an exception caused by one) by `.code`; None otherwise."""
    if genai_errors is None:
        return None
    api_err = exc if isinstance(exc, genai_errors.APIError) else exc.__cause__
    if not isinstance(api_err, genai_errors.APIError):
        return None

    code = api_err.code
    if code == 429:
        retry_after = None
        response = getattr(api_err, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                retry_after = float(headers.get("retry-after"))
            except (TypeError, ValueError):
                retry_after = None
        return AgentErrorDetails(
            code=ProviderErrorCode.RATE_LIMIT,
            provider=provider,
            message="API rate limit exceeded",
            retry_after=retry_after,
        )
    # The Gemini API answers an invalid key with 400 INVALID_ARGUMENT / reason API_KEY_INVALID.
    if code in {401, 403} or (code == 400 and "API_KEY_INVALID" in str(api_err)):
        return AgentErrorDetails(
            code=ProviderErrorCode.AUTH_ERROR,
            provider=provider,
            message="API authentication failure (invalid or forbidden API key)",
        )
    if code == 404:
        return AgentErrorDetails(
            code=ProviderErrorCode.MODEL_NOT_FOUND,
            provider=provider,
            message="Configured model was not found or has been deprecated by provider",
        )
    if isinstance(code, int) and code >= 500:
        return AgentErrorDetails(
            code=ProviderErrorCode.SERVICE_UNAVAILABLE,
            provider=provider,
            message=f"Provider server error (HTTP {code})",
        )
    return None

def classify_provider_error(exc: Exception, provider: str = "unknown") -> AgentErrorDetails:
    """
    Classify third-party provider exceptions (Groq, LiteLLM, Gemini, Google, asyncio)
    into standard structured AgentErrorDetails with clean actionable messages.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return AgentErrorDetails(
            code=ProviderErrorCode.TIMEOUT,
            provider=provider,
            message="Operation timed out (exceeded deadline)"
        )

    # Extract provider or status_code if attached to exception
    prov = getattr(exc, "provider", None) or provider
    status_code = getattr(exc, "status_code", None)

    # Our own daily spend guards (TDD §11): refused before calling the provider.
    if isinstance(exc, (EvidenceBudgetExceeded, LLMBudgetExceeded)):
        return AgentErrorDetails(
            code=ProviderErrorCode.QUOTA_EXHAUSTED,
            provider=prov,
            message=str(exc) or "Daily call budget exhausted"
        )

    # Check for unconfigured service errors
    if "not configured" in str(exc).lower():
        return AgentErrorDetails(
            code=ProviderErrorCode.AUTH_ERROR,
            provider=prov,
            message=f"{_display_name(prov)} API key is not configured"
        )

    # google-genai APIError (Gemini grounded search, Plan 6), raised directly or wrapped as
    # the cause of an EvidenceAPIError. Classified by its HTTP `.code`.
    genai_details = _classify_genai_api_error(exc, prov)
    if genai_details is not None:
        return genai_details

    # LiteLLM / LLM Provider Exceptions
    if litellm is not None:
        if isinstance(exc, litellm.RateLimitError):
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is None:
                # Attempt to extract "Please try again in X.XXs" from error string
                err_str = str(exc)
                match = re.search(r"try again in ([0-9.]+)s", err_str, re.IGNORECASE)
                if match:
                    try:
                        retry_after = float(match.group(1))
                    except ValueError:
                        retry_after = None
            return AgentErrorDetails(
                code=ProviderErrorCode.RATE_LIMIT,
                provider=prov,
                message="Model rate limit reached (requests or tokens per minute)",
                retry_after=retry_after
            )
        if isinstance(exc, litellm.NotFoundError):
            return AgentErrorDetails(
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=prov,
                message="Configured model was not found or has been deprecated by provider"
            )
        if isinstance(exc, litellm.AuthenticationError):
            return AgentErrorDetails(
                code=ProviderErrorCode.AUTH_ERROR,
                provider=prov,
                message="Authentication failed for model provider (check API key)"
            )
        if isinstance(exc, (litellm.APIConnectionError, litellm.ServiceUnavailableError)):
            return AgentErrorDetails(
                code=ProviderErrorCode.SERVICE_UNAVAILABLE,
                provider=prov,
                message="Model provider service is currently unavailable or network failed"
            )

    # HTTP Status inspection (direct or via EvidenceAPIError)
    if status_code is not None:
        if status_code == 432:
            return AgentErrorDetails(
                code=ProviderErrorCode.QUOTA_EXHAUSTED,
                provider=prov,
                message="Provider monthly search quota exhausted"
            )
        elif status_code == 429:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_header = exc.response.headers.get("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        retry_after = None
            return AgentErrorDetails(
                code=ProviderErrorCode.RATE_LIMIT,
                provider=prov,
                message="API rate limit exceeded",
                retry_after=retry_after
            )
        elif status_code in {401, 403}:
            return AgentErrorDetails(
                code=ProviderErrorCode.AUTH_ERROR,
                provider=prov,
                message="API authentication failure (invalid or forbidden API key)"
            )
        elif status_code >= 500:
            return AgentErrorDetails(
                code=ProviderErrorCode.SERVICE_UNAVAILABLE,
                provider=prov,
                message=f"Provider server error (HTTP {status_code})"
            )

    # Search for HTTP status in error string if not directly attached
    status_match = re.search(r"HTTP\s+(\d+)", str(exc))
    if status_match:
        code_int = int(status_match.group(1))
        if code_int == 432:
            return AgentErrorDetails(
                code=ProviderErrorCode.QUOTA_EXHAUSTED,
                provider=prov,
                message="Provider monthly search quota exhausted"
            )
        elif code_int == 429:
            return AgentErrorDetails(
                code=ProviderErrorCode.RATE_LIMIT,
                provider=prov,
                message="API rate limit exceeded"
            )
        elif code_int in {401, 403}:
            return AgentErrorDetails(
                code=ProviderErrorCode.AUTH_ERROR,
                provider=prov,
                message="API authentication failure (invalid or forbidden API key)"
            )

    # HTTP Client Exceptions (Google APIs, Supabase)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 432:
            return AgentErrorDetails(
                code=ProviderErrorCode.QUOTA_EXHAUSTED,
                provider=prov,
                message="Provider monthly search quota exhausted"
            )
        elif status == 429:
            retry_after = None
            retry_header = exc.response.headers.get("retry-after")
            if retry_header:
                try:
                    retry_after = float(retry_header)
                except ValueError:
                    retry_after = None
            return AgentErrorDetails(
                code=ProviderErrorCode.RATE_LIMIT,
                provider=prov,
                message="API rate limit exceeded",
                retry_after=retry_after
            )
        elif status in {401, 403}:
            return AgentErrorDetails(
                code=ProviderErrorCode.AUTH_ERROR,
                provider=prov,
                message="API authentication failure (invalid or forbidden API key)"
            )
        elif status == 404:
            return AgentErrorDetails(
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=prov,
                message="Requested API endpoint or resource was not found"
            )
        elif status >= 500:
            return AgentErrorDetails(
                code=ProviderErrorCode.SERVICE_UNAVAILABLE,
                provider=prov,
                message=f"Provider server error (HTTP {status})"
            )

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return AgentErrorDetails(
            code=ProviderErrorCode.SERVICE_UNAVAILABLE,
            provider=prov,
            message="Connection to provider timed out or failed"
        )

    # General / Parse Exceptions
    if isinstance(exc, (ValueError, KeyError)) and "JSON" in str(exc):
        return AgentErrorDetails(
            code=ProviderErrorCode.PARSE_ERROR,
            provider=prov,
            message="Failed to parse structured model response"
        )

    return AgentErrorDetails(
        code=ProviderErrorCode.UNKNOWN,
        provider=prov,
        message=str(exc) or "Unknown provider error occurred"
    )
