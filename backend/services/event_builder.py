from datetime import datetime, timezone
from uuid import uuid4

from backend.schemas.event import TransitEvent


def create_pothole_event(
    confidence: float,
    bbox: list[float],
    bus_id: str = "BUS-TEST-001",
    camera_id: str = "FRONT-01",
    latitude: float = 0.0,
    longitude: float = 0.0,
) -> TransitEvent:
    return TransitEvent(
        event_id=f"EVT-{uuid4().hex[:8].upper()}",
        event_type="POTHOLE",
        confidence=confidence,
        bus_id=bus_id,
        camera_id=camera_id,
        latitude=latitude,
        longitude=longitude,
        timestamp=datetime.now(timezone.utc),
        bbox=bbox,
        metadata={
            "source": "RF-DETR-S",
            "model": "pothole_rfdetr_s",
        },
    )