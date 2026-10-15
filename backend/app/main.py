import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.services.bias_lookup import load_bias_ratings
from app.agents.linguistic_agent import get_spacy_nlp
from app.services.embedding import generate_embedding
from app.services.nli import get_nli_model
from app.api.routes import router as api_router, limiter
from app.agents.orchestrator import (
    InvalidInputError, NonVerifiableClaimError, AllAgentsFailedError, drain_background_tasks,
    get_verification_runner,
)

logger = logging.getLogger(__name__)
# ADK logs full LLM responses (grounded answer text included) at DEBUG; D4 keeps that text out of logs.
logging.getLogger("google_adk").setLevel(logging.INFO)

SHUTDOWN_DRAIN_TIMEOUT_S = 5.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload heavy assets once at startup per CLAUDE.md Section 7
    logger.info("Initializing VeriFact AI application lifecyle...")
    
    # 0. R8.4: an absent evidence key is now an outage (EvidenceAPIError), not a silent
    # empty result. Say so once at startup so a mis-configured deployment is obvious
    # instead of showing up as every claim degrading to "Partial Analysis". Non-fatal.
    missing_keys = [
        name for name, value in (
            ("GEMINI_API_KEY", settings.gemini_api_key),
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
        
    # 4. Preload NLI cross-encoder (source stance + cache equivalence)
    logger.info("Preloading NLI model...")
    try:
        get_nli_model()
    except Exception as exc:
        logger.warning(f"NLI model preload warning: {exc}")

    # 5. Build the ADK verification workflow + Runner. Not wrapped: a graph-configuration
    # error must fail at boot, not on the first request.
    logger.info("Building ADK verification workflow...")
    get_verification_runner()

    logger.info("Startup complete. Pipeline is ready.")
    yield
    logger.info("Shutting down VeriFact AI application...")
    # Let cache writes that outlived their request finish instead of dropping them.
    await drain_background_tasks(timeout=SHUTDOWN_DRAIN_TIMEOUT_S)

app = FastAPI(
    title="VeriFact AI Verification Pipeline API",
    version="0.2.0",
    description="Multi-agent claim verification pipeline with transparent composite trust score.",
    lifespan=lifespan
)

# Attach slowapi rate limiter state
app.state.limiter = limiter

# CORS middleware for frontend integration (Vite + React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this, browsers hide the latency header from the cross-origin frontend.
    expose_headers=["X-Process-Time-Ms"],
)

@app.middleware("http")
async def process_time_header(request: Request, call_next):
    """H1 / PRD §10: per-request latency, logged for p95 and returned to the client."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.info("request_timing path=%s status=%s ms=%d", request.url.path, response.status_code, elapsed_ms)
    return response

# Custom exception handlers for pipeline errors
@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    # Schema-level rejection (missing field, >10000-char body). Distinct from an opinion (TDD §3.1).
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid Request", "code": "request_validation", "detail": str(exc.errors()[:3])},
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limited_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate Limited", "code": "rate_limited", "detail": str(exc.detail)},
    )

@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid Input", "code": "invalid_input", "detail": str(exc)}
    )

@app.exception_handler(NonVerifiableClaimError)
async def non_verifiable_handler(request: Request, exc: NonVerifiableClaimError):
    return JSONResponse(
        status_code=422,
        content={"error": "Non-Verifiable Claim", "code": "non_verifiable_claim", "detail": str(exc)}
    )

@app.exception_handler(AllAgentsFailedError)
async def all_agents_failed_handler(request: Request, exc: AllAgentsFailedError):
    return JSONResponse(
        status_code=500,
        content={"error": "Verification Failed", "code": "all_agents_failed", "detail": str(exc)}
    )

# Include API routes
app.include_router(api_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "verifact-ai"}
