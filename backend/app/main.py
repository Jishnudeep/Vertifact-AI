from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.bias_lookup import load_bias_ratings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load bias ratings at startup
    load_bias_ratings()
    yield

app = FastAPI(title="Verifact AI Backend API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

