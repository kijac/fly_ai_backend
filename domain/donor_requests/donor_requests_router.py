from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from database import get_db

from domain.donor_requests import donor_requests_schema, donor_requests_crud
from model import Donor_Requests

router = APIRouter(
    prefix="/api/donor_requests",
)


# 장난감을 신청받고 db 저장하는 api
# 크레딧 확인 -> 요청 생성 -> 아이템 재고 예약
@router.post("/request_delivery", response_model=donor_requests_schema.RequestDeliveryResponse, status_code=status.HTTP_201_CREATED)
async def request_delivery(
    body: donor_requests_schema.RequestDeliveryBody,
    db: Session = Depends(get_db),
):
    req: Donor_Requests = await donor_requests_crud.create_donor_request_points_only(
        db = db,
        recipient = body.recipient_id,
        toy_ids = body.toy_ids,
        note = body.note,
        )

    recipient_status = req.recipient_status.value if hasattr(req.recipient_status, "value") else req.recipient_status
    delivery_status = req.delivery_status.value if hasattr(req.delivery_status, "value") else req.delivery_status

    return donor_requests_schema.RequestDeliveryResponse(
        request_id = req.id,
        recipient_id = req.recipient_id,
        items = [donor_requests_schema.RequestDeliveryItem(toy_id=t.toy_stock_id) for t in req.request_items],
        recipient_status = recipient_status,
        delivery_status = delivery_status,
    )

# 배송내역 조회시 리스트를 보내주는 api
# @router.get("/orders")