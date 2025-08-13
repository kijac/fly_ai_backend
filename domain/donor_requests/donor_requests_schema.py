from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, conlist

class RequestDeliveryBody(BaseModel):
    recipient_id: int = Field(..., ge=1)
    toy_ids: Annotated[list[int], Field(min_items=1)]
    note: Optional[str] = Field(None, max_length=500)

class RequestDeliveryItem(BaseModel):
    toy_id: int

class RequestDeliveryResponse(BaseModel):
    request_id: int
    recipient_id: int
    items: List[RequestDeliveryItem]
    recipient_status: str
    delivery_status: str