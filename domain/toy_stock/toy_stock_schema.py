from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from model import DonationStatus, ToyStatus

class Toy(BaseModel):
    toy_id: int
    toy_name: str
    toy_type: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    sale_price: int

    class Config:
        from_attributes = True

class ToyDetail(BaseModel):
    toy_id: int
    user_id: int 
    toy_name: Optional[str] = None
    toy_type: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None
    is_donatable: Optional[DonationStatus] = None   # Enum → str로 변환
    reject_reason: Optional[str] = None
    glb_model_url: Optional[str] = None
    toy_status: Optional[ToyStatus] = None   # Enum → str
    reserved_by_request_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    sale_price: Optional[int] = None
    purchase_price: Optional[int] = None
    sale_status: Optional[str] = None

    class Config:
        from_attributes = True
        use_enum_values = True


class ToyStockList(BaseModel):
    total: int = 0
    toystock_list: list[Toy] = []
        