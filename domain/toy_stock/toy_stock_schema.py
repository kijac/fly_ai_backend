from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from model import DonationStatus, ToyStatus

class Toy(BaseModel):
    toy_id: int
    toy_name: str
    toy_type: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # kim의 구조화된 이미지 URL
    description: Optional[str] = None
    sale_price: int

    class Config:
        from_attributes = True

class ToyDetail(BaseModel):
    toy_id: int
    user_id: int  # donor_id에서 user_id로 변경
    toy_name: Optional[str] = None
    toy_type: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None  # kim의 구조화된 이미지 URL
    is_donatable: Optional[DonationStatus] = None   # Enum 타입 유지
    reject_reason: Optional[str] = None
    glb_model_url: Optional[str] = None
    toy_status: Optional[ToyStatus] = None     # Enum 타입 유지
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
        

class SaleList(BaseModel):
    toy_id: int
    toy_name: Optional[str]
    toy_type: Optional[str]
    image_url: Optional[List[str]]  # JSON 배열로 저장
    reject_reason: Optional[str]
    toy_status: Optional[str]
    created_at: datetime
    purchase_price: Optional[int]

class ToySaleDetail(BaseModel):
    """판매용 장난감 상세 정보 스키마"""
    toy_id: int
    user_id: int
    toy_name: Optional[str] = None
    toy_type: Optional[str] = None
    image_url: Optional[List[str]] = None  # JSON 배열로 저장
    toy_status: Optional[str] = None
    sale_price: Optional[int] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ToySaleResponse(BaseModel):
    """판매 등록 응답 스키마"""
    success: bool
    message: str
    registered_count: int
    points_added: int
    current_points: int

class ToyDonationDetail(BaseModel):
    """기부용 장난감 상세 정보 스키마"""
    toy_id: int
    user_id: int
    toy_name: Optional[str] = None
    toy_type: Optional[str] = None
    image_url: Optional[List[str]] = None  # JSON 배열로 저장
    toy_status: Optional[str] = None
    is_donatable: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ToyDonationResponse(BaseModel):
    """기부 등록 응답 스키마"""
    success: bool
    message: str
    registered_count: int
    points_added: int
    current_points: int