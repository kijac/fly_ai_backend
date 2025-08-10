from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Toy(BaseModel):
    toy_id: int
    toy_name: str
    toy_type: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None

    class Config:
        orm_mode = True

class ToyDetail(BaseModel):
    toy_id: int
    donor_id: int
    toy_name: Optional[str] = None
    toy_type: Optional[str] = None
    image_url: Optional[str] = None
    is_donatable: Optional[str] = None   # Enum → str로 변환
    reject_reason: Optional[str] = None
    glb_model_url: Optional[str] = None
    donor_status: Optional[str] = None   # Enum → str
    reserved_by_request_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    description: Optional[str] = None

    class Config:
        orm_mode = True


class ToyStockList(BaseModel):
    total: int = 0
    toystock_list: list[Toy] = []
        

class DonationList(BaseModel):
    toy_type: Optional[str]
    donor_status: Optional[str]
    created_at: datetime
    image_url: Optional[str]
    has_glb_model: bool