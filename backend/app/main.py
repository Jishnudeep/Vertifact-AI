import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.services.bias_lookup import load_bias_ratings
from app.agents.linguistic_agent import get_spacy_nlp
from app.services.embedding import generate_embedding
from app.api.routes import router as api_router, limiter
from app.agents.orchestrator import InvalidInputError, NonVerifiableClaimError, AllAgentsFailedError

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload heavy assets once at startup per CLAUDE.md Section 7
    logger.info("Initializing VeriFact AI application lifecyle...")
    
    # 0. R8.4: an absent evidence key is now an outage (EvidenceAPIError), not a silent
    # empty result. Say so once at startup so a mis-configured deployment is obvious
    # instead of showing up as every claim degrading to "Partial Analysis". Non-fatal.
    missing_keys = [
        name for name, value in (
            ("TAVILY_API_KEY", settings.tavily_api_key),
            ("GOOGLE_FACTCHECK_API_KEY", settings.google_fact_check_api_key),
            ("GROQ_API_KEY", settings.groq_api_key),
        )
        if not value
    ]
    if missing_keys:
        logger.warning(
            "Evidence/model API keys are NOT configured: %s. "
            "Agents depending on them will degrade to 'Partial Analysis' on every request.",
            ", ".join(missing_keys),
        )

    # 1. Load AllSides bias ratings
    logger.info("Loading AllSides bias ratings...")
    load_bias_ratings()
    
    # 2. Preload spaCy NLP pipeline
    logger.info("Preloading spaCy 'en_core_web_sm'...")
    try:
        get_spacy_nlp()
    except Exception as exc:
        logger.warning(f"spaCy preload warning: {exc}")
        
    # 3. Preload SentenceTransformer embedding model
    logger.info("Preloading embedding model...")
    try:
        generate_embedding("VeriFact AI pipeline warmup embedding.")
    except Exception as exc:
        logger.warning(f"Embedding model preload warning: {exc}")
        
    logger.info("Startup complete. Pipeline is ready.")
    yield
    logger.info("Shutting down VeriFact AI application...")

app = FastAPI(
    title="VeriFact AI Verification Pipeline API",
    version="0.2.0",
    description="Multi-agent claim verification pipeline with transparent composite trust score.",
    lifespan=lifespan
)

# Attach slowapi rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for frontend integration (Vite + React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handlers for pipeline errors
@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid Input", "detail": str(exc)}
    )

@app.exception_handler(NonVerifiableClaimError)
async def non_verifiable_handler(request: Request, exc: NonVerifiableClaimError):
    return JSONResponse(
        status_code=422,
        content={"error": "Non-Verifiable Claim", "detail": str(exc)}
    )

@app.exception_handler(AllAgentsFailedError)
async def all_agents_failed_handler(request: Request, exc: AllAgentsFailedError):
    return JSONResponse(
        status_code=500,
        content={"error": "Verification Failed", "detail": str(exc)}
    )

# Include API routes
app.include_router(api_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "verifact-ai"}
