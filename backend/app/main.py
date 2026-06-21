from fastapi import FastAPI

app = FastAPI(title="Verifact AI Backend API")

@app.get("/health")
async def health_check():
    return {"status": "ok"}
