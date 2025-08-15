from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session
from database import get_db
from model import DeliveryStatus
from typing import Optional
from domain.donor_requests import donor_requests_schema, donor_requests_crud
from model import Donor_Requests

router = APIRouter(
    prefix="/api/donor_requests",
)

# 장난감을 구매 신청하고 db 저장하는 api
# 요청 생성 -> 아이템 재고 예약
@router.post("/request_delivery", response_model=donor_requests_schema.RequestDeliveryResponse, status_code=status.HTTP_201_CREATED)
def request_delivery(
    body: donor_requests_schema.RequestDeliveryBody,
    db: Session = Depends(get_db),
):
    req: Donor_Requests = donor_requests_crud.create_donor_request(
        db = db,
        recipient_id = body.recipient_id,
        toy_ids = body.toy_ids,
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



# 구매내역 조회시 리스트를 보내주는 api
@router.get("/orders", response_model=donor_requests_schema.PurchaseHistoryList, status_code=status.HTTP_200_OK)
def get_purchase_history(
    user_id: int,
    page: int = Query(0, ge=0, description="페이지 번호(0부터)"),
    size: int = Query(10, ge=1, le=50, description="페이지당 아이템 수(1~50)"),
    status_: Optional[DeliveryStatus] = Query(
        None, alias="status", description="배송상태(PREPARING, ONGOING, COMPLETED)"),
    db: Session = Depends(get_db),
    ):
    """
    구매/배송 내역 조회:
    - 마스터: donor_requests (recipient_id = user_id)
    - 필터: delivery_status (옵션)
    - 정렬: 최근 요청 우선 (requested_at DESC)
    - 응답: request_id, delivery_status, total_price, requested_at, toy_ids[]
    """
    total, reqs = donor_requests_crud.get_user_order_history(
        db = db,
        user_id = user_id,
        page = page,
        size = size,
        status_ = status_,
    )

    orders = []
    for r in reqs:
        delivery_status = r.delivery_status.value if hasattr(r.delivery_status, "value") else r.delivery_status
        orders.append({
            "request_id": r.id,
            "recipient_id": r.recipient_id,
            "delivery_status": delivery_status,
            "total_price": r.total_price,
            "requested_at": r.requested_at,
            "items": [
                donor_requests_schema.PurchaseHistoryItem(
                    toy_id=t.toy_stock_id,
                    toy_name=(t.toy_stock.toy_name if t.toy_stock else None),
                    unit_price=t.unit_price,
                ) for t in (r.request_items or [])
            ]
        })

    return donor_requests_schema.PurchaseHistoryList(
        total=total,
        orders=orders,
    )