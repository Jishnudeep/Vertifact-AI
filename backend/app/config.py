import os
from pathlib import Path
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    supabase_url: str = Field(validation_alias=AliasChoices("SUPABASE_URL"))
    supabase_publish_key: str = Field(validation_alias=AliasChoices("SUPABASE_PUBLISH_KEY"))
    supabase_secret_key: str = Field(validation_alias=AliasChoices("SUPABASE_SECRET_KEY"))
    groq_api_key: str = Field(validation_alias=AliasChoices("GROQ_API_KEY"))
    tavily_api_key: str = Field(validation_alias=AliasChoices("TAVILY_API_KEY", "TAVILY_KEY"))
    google_fact_check_api_key: str = Field(validation_alias=AliasChoices("GOOGLE_FACTCHECK_API_KEY", "GOOGLE_FACT_CHECK_API_KEY"))
    
    # Model ID comes from config as per AGENTS.md Section 3
    model_id: str = "groq/llama-3.1-8b-instant"
    
    # Model name for generating embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Dimension of the embeddings used (384 for MiniLM in dev, 1024 for BGE-M3 in prod)
    embedding_dimension: int = 384
    
    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent / ".env"),
            str(Path(__file__).resolve().parent.parent / ".env"),
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Module-level variables for clean backward compatibility
SUPABASE_URL = settings.supabase_url
SUPABASE_KEY = settings.supabase_secret_key
GROQ_API_KEY = settings.groq_api_key
TAVILY_API_KEY = settings.tavily_api_key
GOOGLE_FACT_CHECK_API_KEY = settings.google_fact_check_api_key
