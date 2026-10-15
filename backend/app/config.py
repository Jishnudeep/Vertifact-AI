import os
from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    supabase_url: str = Field(default="", validation_alias=AliasChoices("SUPABASE_URL"))
    supabase_publish_key: str = Field(default="", validation_alias=AliasChoices("SUPABASE_PUBLISH_KEY"))
    supabase_secret_key: str = Field(default="", validation_alias=AliasChoices("SUPABASE_SECRET_KEY"))
    groq_api_key: str = Field(default="", validation_alias=AliasChoices("GROQ_API_KEY"))
    google_fact_check_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_FACTCHECK_API_KEY", "GOOGLE_FACT_CHECK_API_KEY"))
    # Plan 6 D1: Gemini API (AI Studio) key for the grounded-search agents. Not Vertex AI.
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"))

    # Allowed CORS origins for frontend integration (Vite dev server default 5173)
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Model ID comes from config as per CLAUDE.md §3, TDD §4.1, PRD §5 — never hardcoded in agents.
    #
    # Dev model: `groq/openai/gpt-oss-120b`. This deviates from the originally documented
    # `groq/llama-3.1-8b-instant` because Groq no longer serves ANY Llama chat model on the
    # developer tier — those moved to Enterprise-tier access (contact sales), they are not
    # deprecated. Verified live 2026-09-10 against this project's Groq key:
    #   groq/llama-3.1-8b-instant    -> 404 NotFoundError: GroqException - {"error":{"message":
    #     "The model `llama-3.1-8b-instant` does not exist or you do not have access to it.",
    #     "type":"invalid_request_error","code":"model_not_found"}}
    #   groq/llama-3.3-70b-versatile -> 404, identical model_not_found body
    #   groq/openai/gpt-oss-120b     -> 200, clean JSON on both the verdict-extraction and the
    #                                   claim-alignment prompts (reasoning in a separate field)
    # GET https://api.groq.com/openai/v1/models on the same key lists exactly:
    #   allam-2-7b, groq/compound, groq/compound-mini, llama-prompt-guard-2-22m,
    #   llama-prompt-guard-2-86m, openai/gpt-oss-120b, openai/gpt-oss-20b,
    #   openai/gpt-oss-safeguard-20b, qwen/qwen3.6-27b, qwen/qwen3.8-27b, whisper-*
    # — no Llama chat model, so no Llama-family fallback is reachable.
    # Model card: https://console.groq.com/docs/models (gpt-oss-120b is a PRODUCTION model,
    # ~500 tok/s, $0.15 / $0.60 per M input/output tokens).
    # 120b over 20b: stronger adherence to the strict JSON contract on the alignment and
    # verdict-extraction prompts, which is what the response parser depends on.
    # It is a reasoning model, so max_tokens is 1024 in adk_helpers.py — 256 truncated the JSON
    # body and forced the (injection-prone) regex fallback (R7.B1/R7.P4).
    # Revisit when prod moves to vLLM (PRD §5).
    model_id: str = "groq/openai/gpt-oss-120b"

    # Plan 6: Gemini model for the `google_search` grounded-search agents (ADK's google_search
    # tool only runs on Gemini models). UNVERIFIED: the Task 0 live probe was declined, so this
    # id has not been confirmed against the project key — it is checked in the Task 9 live gate.
    # Override with SEARCH_MODEL_ID.
    search_model_id: str = "gemini-3.1-flash-lite"

    # Model name for generating embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Dimension of the embeddings used (384 for MiniLM in dev, 1024 for BGE-M3 in prod)
    embedding_dimension: int = 384
    
    # Local NLI cross-encoder for source stance (Cross-Ref agent) and claim equivalence
    # (semantic cache). Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md D1.
    # Apache-2.0, ~0.1B params, CPU. Loaded once in main.py lifespan.
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"

    # Token budget per (premise, hypothesis) pair; longer snippets are truncated by the tokenizer.
    nli_max_length: int = 256

    # Semantic cache TTLs (C2). A matched fact-check changes slowly; everything else is
    # news-driven and may be overtaken by new reporting within a day.
    cache_ttl_fact_checked_hours: int = 720
    cache_ttl_default_hours: int = 24
    
    # H1: total wall-clock budget per /api/verify request (PRD §10: < 10 s end-to-end).
    request_deadline_s: float = 9.5
    
    # Groq free tier for gpt-oss-120b = 1,000 requests/day; keep headroom for retries.
    llm_daily_call_budget: int = 900
    # Plan 6: Gemini grounded-search calls per UTC day (one per Fact-Check / Cross-Ref run).
    search_daily_call_budget: int = 50

    # Plan 6 / TDD §11: overall wall-clock budget for fetching one cited page (all hops).
    page_fetch_timeout_s: float = 2.0

    # H3: number of reverse proxies in front of the API that append to X-Forwarded-For.
    # 0 = trust nothing (local dev); 1 = one platform proxy (Render/Railway/Fly).
    trusted_proxy_hops: int = 0
    
    # H4: claims/results older than this are deleted by purge_old_claims (migration 05).
    retention_days: int = 90
    
    # Linguistic agent LLM sensationalism flag (Build Order 1.2 default is False: heuristics only)
    enable_llm_sensationalism: bool = False
    
    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent / ".env"),
            str(Path(__file__).resolve().parent.parent / ".env"),
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

def _export_provider_env(s: Settings) -> None:
    """
    Export keys that SDKs read from os.environ rather than from our Settings.

    google-genai (used by ADK's Gemini model for the grounded-search agents) reads
    GOOGLE_API_KEY. It is assigned, not setdefault, so the SDK uses exactly the key Settings
    resolved. Key values are never logged.

    The backend selector defaults to "false" so the Gemini API is used, not Vertex AI
    (Plan 6 D1); an explicit operator setting is left alone. google-genai 2.x / ADK 2.12 read
    GOOGLE_GENAI_USE_ENTERPRISE first and warn that GOOGLE_GENAI_USE_VERTEXAI is deprecated
    when only the old name is present, so both are set, to the same value.
    """
    if s.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = s.gemini_api_key
    backend_selector = (
        os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        or "false"
    )
    os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", backend_selector)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", backend_selector)


@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    _export_provider_env(s)
    return s

def __getattr__(name: str):
    """
    Lazy accessor for settings and backward-compatible module globals.
    Defers Settings initialization until accessed.
    """
    if name == "settings":
        return get_settings()
    s = get_settings()
    mapping = {
        "SUPABASE_URL": s.supabase_url,
        "SUPABASE_KEY": s.supabase_secret_key,
        "GROQ_API_KEY": s.groq_api_key,
        "GEMINI_API_KEY": s.gemini_api_key,
        "GOOGLE_FACT_CHECK_API_KEY": s.google_fact_check_api_key,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
