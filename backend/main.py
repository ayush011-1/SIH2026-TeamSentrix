from fastapi import FastAPI

from backend.api.events import router as events_router


app = FastAPI(
    title="TransitEye Backend",
    version="1.0.0",
)

app.include_router(events_router)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "service": "TransitEye Backend",
    }