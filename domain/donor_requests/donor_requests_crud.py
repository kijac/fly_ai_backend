from typing import List
from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from model import DonationStatus, Toy_Stock, Donor_Requests, Donor_Requests_Item, User, RecipientStatus, DeliveryStatus

# 내부 포인트로 승인 (User.points >= 합계)
def authorize_credit_points(db: Session, user_id: int, toy_ids: list[int]) -> bool:
    toy_ids = list(set(toy_ids)) # 중복 방지

    total_amount = db.execute(
        select(func.coalesce(func.sum(Toy_Stock.price), 0)).where(Toy_Stock.toy_id.in_(toy_ids))
    ).scalar_one_or_none()
    
    user_points = db.execute(
        select(func.coalesce(func.sum(User.points), 0)).where(User.user_id == user_id)
    ).scalar_one_or_none()

    return user_points >= total_amount

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
def validate_toys(toys: List[Toy_Stock]) -> None:
    for t in toys:
        if t.is_donatable not in (DonationStatus.RECYCLABLE, DonationStatus.UPCYCLE):
            raise HTTPException(status_code=422, detail=f"toy {t.toy_id} is not donatble")
        if t.reserved_by_request_id is not None:
            raise HTTPException(status_code=409, detail=f"toy {t.toy_id} is already reserved")

# 마스터 생성 (Donor_Requests 한 행 생성)
def create_request_master(db: Session, recipient_id: int) -> Donor_Requests:
    req = Donor_Requests(
        recipient_id=recipient_id,
        recipient_status=RecipientStatus.PENDING,
        delivery_status=DeliveryStatus.PREPARING,
        requested_at=func.now(),
        created_at=func.now(),
    )
    db.add(req)
    db.flush()
    return req

# 아이템 생성
def create_request_items(db: Session, req_id: int, toys: List[Toy_Stock]) -> None:
    for t in toys:
        db.add(Donor_Requests_Item(requested_id=req_id, toy_stcok_id=t.toy_id))
    db.flush()

# 재고 예약 표시
def mark_toys_reserved(db: Session, req_id: int, toys: List[Toy_Stock]) -> None:
    for t in toys:
        t.reserved_by_request_id = req_id
    db.flush()

# 전체 플로우 (트랜잭션, 내부 포인트 승인 + 차감/홀드까지)
def create_donor_request_points_only(
    db: Session,
    recipient_id: int,
    toy_ids: List[int],
    note: str | None,) -> Donor_Requests:
    if not toy_ids:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="toy_ids is empty")

    toy_ids = list(set(toy_ids)) # 중복 제거

    # 0) 승인(읽기 전용): 포인트 차감/홀드 없음
    ok = authorize_credit_points(db, recipient_id, toy_ids)
    if not ok:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="insufficient points")

    # 1) 트랜잭션: 재고만 잠그고 생성/예약
    try:
        with db.begin():
            # 재고 검증 + 검증
            toys = lock_and_load_toys(db, toy_ids)
            validate_toys(toys)

            # 마스터 생성
            req = Donor_Requests(
                recipient_id=recipient_id,
                recipient_status=RecipientStatus.PENDING,
                delivery_status=DeliveryStatus.PREPARING,
                requested_at=func.now(),
                created_at=func.now(),
            )
            db.add(req)
            db.flush()

            # 아이템 생성 + 재고 예약
            for t in toys:
                db.add(Donor_Requests_Item(requested_id=req.id, toy_stock_id=t.toy_id))
                t.reserved_by_request_id = req.id

            db.refresh(req)
            return req

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to create donor request") from e
            
