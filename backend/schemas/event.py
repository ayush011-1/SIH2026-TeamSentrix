from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransitEvent(BaseModel):
    event_id: str

    event_type: Literal[
        "POTHOLE",
        "ROAD_DAMAGE",
        "WATERLOGGING",
        "TRAFFIC_INCIDENT",
        "ANPR",
        "OTHER",
    ]

    confidence: float = Field(ge=0.0, le=1.0)

    bus_id: str
    camera_id: str

    latitude: float
    longitude: float

    timestamp: datetime

    image_path: Optional[str] = None

    bbox: Optional[list[float]] = None

    metadata: Optional[dict] = None