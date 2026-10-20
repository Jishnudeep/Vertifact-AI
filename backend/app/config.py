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
    
    # Model of every worker LlmAgent and of the grounded-search sub-agent (ADK's native Gemini,
    # AI Studio key; never hardcoded in agent logic). Groq/gpt-oss is tabled for now.
    # Override with MODEL_ID / SEARCH_MODEL_ID.
    model_id: str = "gemini-3.1-flash-lite"
    # Gemini model for the `google_search` grounded-search sub-agent (it can't be a tool of a
    # model that also has function tools, so the search tools call it as a service).
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
    
    # H1: total wall-clock budget per /api/verify request. The PRD §10 "< 10 s" target and the
    # 8 s per-agent limit are no longer design constraints: the agents are real ADK tool loops
    # whose latency is unknown until measured. These are generous BACKSTOPS, to be replaced by
    # the values the benchmark-then-optimize phase sets (eval/latency_bench.py).
    request_deadline_s: float = 35.0
    # Backstop for one worker agent (all of its LLM turns and tool calls). TBD from benchmarking.
    agent_timeout_s: float = 30.0

    # Daily cap on worker LLM turns (a tool loop is several turns per agent).
    llm_daily_call_budget: int = 900
    # Plan 6: Gemini grounded-search calls per UTC day. One uncached request spends up to 3: one
    # Fact-Check web search plus up to two Cross-Ref coverage searches. 150 therefore allows about
    # 50 uncached requests a day at worst. TBD from benchmarking (eval/latency_bench.py reports the
    # real searches per request) and from the provider's actual grounding quota.
    search_daily_call_budget: int = 150

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
