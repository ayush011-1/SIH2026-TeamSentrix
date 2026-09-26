from fastapi import APIRouter

from backend.schemas.event import TransitEvent

router = APIRouter(prefix="/api/events", tags=["Events"])

EVENTS: list[TransitEvent] = []


@router.post("")
async def ingest_event(event: TransitEvent):
    EVENTS.append(event)

    return {
        "success": True,
        "message": "Event stored successfully",
        "event": event.model_dump(mode="json"),
        "total_events": len(EVENTS),
    }


@router.get("")
async def get_events():
    return {
        "success": True,
        "total_events": len(EVENTS),
        "events": [
            event.model_dump(mode="json")
            for event in EVENTS
        ],
    }