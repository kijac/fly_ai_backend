from typing import List, Optional, Annotated
from pydantic import BaseModel, Field
from datetime import datetime

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


class PurchaseHistoryItem(BaseModel):
    toy_id: int
    toy_name: str
    unit_price: int

class PurchaseHistoryRow(BaseModel):
    request_id: int
    recipient_id: int
    delivery_status: str
    total_price: int
    requested_at: datetime
    items: List[PurchaseHistoryItem]

    class Config:
        from_attributes = True

class PurchaseHistoryList(BaseModel):
    total: int
    orders: List[PurchaseHistoryRow]