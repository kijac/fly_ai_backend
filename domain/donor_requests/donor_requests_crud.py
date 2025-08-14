from typing import List, Tuple
from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session, selectinload
from model import (
    DonationStatus, Toy_Stock, Donor_Requests, Donor_Requests_Item,\
    User, RecipientStatus, DeliveryStatus, ToyStatus)
from typing import Optional

# 유저 조회
def _get_user_or_404(db: Session, user_id: int) -> User:
    u = db.execute(select(User).where(User.user_id == user_id)).scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return u

# 재고 잠금 + 로드
def lock_and_load_toys(db: Session, toy_ids: List[int]) -> List[Toy_Stock]:
    # 1. 쿼리 작성
    q = (
        select(Toy_Stock)
        .where(Toy_Stock.toy_id.in_(toy_ids))
        .with_for_update()
    )
    # 2. 쿼리 실행 + ORM 객체 변환
    toys = db.execute(q).scalars().all()
    # 3. 누락된 장난감 체크
    found = {t.toy_id for t in toys}
    missing = set(toy_ids) - found
    if missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"재고 조회 실패: 장난감 ID {sorted(list(missing))} 누락")
    return toys

# 유효성 검사
def validate_toys_request(toys: List[Toy_Stock], recipient_id: int) -> None:
    for t in toys:
        # 기부 가능 범주인지 (판매용이면 기부 불가)
        if t.is_donatable not in (DonationStatus.RECYCLABLE, DonationStatus.UPCYCLE):
            raise HTTPException(status_code=422, detail=f"toy {t.toy_id} is not donatble")
        # 이미 예약된 장난감인지
        if t.reserved_by_request_id is not None:
            raise HTTPException(status_code=409, detail=f"toy {t.toy_id} is already reserved")
        # 상태 규칙 (요청 불가 상태 금지)
        if t.toy_status in (ToyStatus.REJECTED, ToyStatus.DONATION, ToyStatus.END_BUY):
            raise HTTPException(status_code=409, detail=f"toy {t.toy_id} is not requestable")
        # 자기 물건 신청 금지 (정책에 맞게 필요 없으면 삭제)
        if t.user_id == recipient_id:
            raise HTTPException(status_code=400, detail=f"toy {t.toy_id} is not requestable")
        # 가격 스냅샷 유효성 (총액 계산/아이템 스냅샷에 사용)
        if t.sale_price is None or int(t.sale_price) <= 0:
            raise HTTPException(status_code=422, detail=f"toy {t.toy_id} has invalid price")

# 마스터 생성 (Donor_Requests 한 행 생성)
def create_request_master(db: Session, recipient_id: int, total_price: int) -> Donor_Requests:
    req = Donor_Requests(
        recipient_id=recipient_id,
        recipient_status=RecipientStatus.PENDING,
        delivery_status=DeliveryStatus.PREPARING,
        requested_at=func.now(),
        total_price=total_price,
    )
    db.add(req)
    db.flush()
    return req

# 아이템 생성
def create_request_items(db: Session, req: Donor_Requests, toys: List[Toy_Stock]) -> None:
    for t in toys:
        req.request_items.append(
            Donor_Requests_Item(
                requested_id=req.id,
                toy_stock_id=t.toy_id,
                unit_price=int(t.sale_price),
            )
        )
    db.flush()

# 재고 예약 표시(구매용 요청)
def mark_toys_reserved(db: Session, req_id: int, toys: List[Toy_Stock]) -> None:
    for t in toys:
        t.reserved_by_request_id = req_id
        t.toy_status = ToyStatus.END_BUY
        t.updated_at = func.now()
    db.flush()

# 전체 플로우 (판매 요청 트랜잭션)
def create_donor_request(
    db: Session,
    recipient_id: int,
    toy_ids: List[int],
) -> Donor_Requests:
    if not toy_ids:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="toy_ids is empty")

    toy_ids = list(set(toy_ids)) # 중복 제거

    try:
        # 0) 수령자 존재 확인(+락)
        _get_user_or_404(db, recipient_id)

        # 1) 재고 검증 + 검증
        toys = lock_and_load_toys(db, toy_ids)
        validate_toys_request(toys, recipient_id)

        # 2) 총액 계산
        total_price = sum(int(t.sale_price) for t in toys)

        # 3) 마스터 생성
        req = create_request_master(db, recipient_id, total_price)
        
        # 4) 아이템 생성 + 재고 예약
        create_request_items(db, req=req, toys=toys)
        mark_toys_reserved(db, req_id=req.id, toys=toys)
        
        # 관계까지 메모리에 보장되게 만들기
        db.refresh(req)
        _ = req.request_items
        return req

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        import logging
        logging.exception("create_donor_request_points_only failed")
        raise HTTPException(status_code=500, detail=str(getattr(e, "orig", e)))
            
def get_user_order_history(
    db: Session,
    user_id: int,
    page: int,
    size: int,
    status_: Optional[DeliveryStatus]) -> Tuple[int, List[Donor_Requests]]:

    q = (
        select(Donor_Requests)
        .where(Donor_Requests.recipient_id == user_id)
        .options(
            selectinload(Donor_Requests.request_items)
            .selectinload(Donor_Requests_Item.toy_stock)
        )
    )
    if status_ is not None:
        q = q.where(Donor_Requests.delivery_status == status_)

    total = db.scalar(
        select(func.count())
        .select_from(Donor_Requests)
        .where(
            Donor_Requests.recipient_id == user_id,
            *( [Donor_Requests.delivery_status == status_] if status_ is not None else [] )
        )
    ) or 0

    rows = (
        db.execute(
            q.order_by(Donor_Requests.requested_at.desc())
            .offset(page * size)
            .limit(size)
        )
        .scalars()
        .all()
    )

    return total, rows