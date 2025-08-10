from fastapi import APIRouter, Depends, HTTPException, Query, Form, File, UploadFile
from sqlalchemy.orm import Session
from starlette import status
from typing import List, Optional
import os
import shutil

from database import get_db
from domain.toy_stock import toy_stock_schema, toy_stock_crud
from fastapi import Form, File, UploadFile
from model import DonationStatus, DonorStatus, User
from datetime import datetime
from domain.user.user_router import get_current_user

router = APIRouter(
    prefix="/api/toy_stock",
)

# 장난감 리스트 불러오기
@router.get("/toystokclist", response_model=toy_stock_schema.ToyStockList)
def toystock_list(
    toy_type: str = Query("", description="장난감 종류 검색 (없으면 전체 조회)"), # Path Parameter
    page: int = Query(0, ge=0, description="페이지 번호 (0부터 시작)"),
    size: int = Query(10, ge=1, le=50, description="한 페이지에 보여줄 개수"),
    keyword: str = Query("", description="검색 키워드 (없으면 전체 조회)"),
    db: Session = Depends(get_db)
    ):
    total, _toystock_list = toy_stock_crud.get_toystock_list(db, toy_type=toy_type, skip=page*size, limit=size, keyword=keyword)
    return {
        "total": total, 
        "toystock_list": _toystock_list
    }

# 장난감 Detail 정보 불러오기
@router.get("/detail/{toy_id}", response_model=toy_stock_schema.ToyDetail)
def toy_detail(toy_id: int, db: Session = Depends(get_db)):
    toy = toy_stock_crud.get_toy(db, toy_id)
    return toy



## 주문한 장난감 리스트 불러오기
#@router.get("/mytoy", status_code=status.HTTP_200_OK)


@router.post("/donation")
async def register_toys_bulk(
    toy_type: List[str] = Form(...),
    is_donatable: List[str] = Form(...),
    description: List[str] = Form(...),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not toy_type or not is_donatable or not images:
        raise HTTPException(status_code=400, detail="필수 항목이 누락되었습니다.")

    if not (len(toy_type) == len(is_donatable) == len(description) == len(images)):
        raise HTTPException(status_code=400, detail="입력 데이터의 개수가 일치하지 않습니다.")

    errors = []
    success_count = 0
    for idx in range(len(toy_type)):
        # ENUM 값 변환 예외처리
        try:
            donation_status = DonationStatus(is_donatable[idx])
        except ValueError:
            errors.append(f"{toy_type[idx]}: is_donatable 값이 올바르지 않습니다.")
            continue

        # 이미지 파일 확장자 체크
        if not images[idx].filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            errors.append(f"{toy_type[idx]}: 이미지 파일만 업로드 가능합니다.")
            continue

        # 이미지 저장
        save_dir = "toypics"
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, images[idx].filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(images[idx].file, buffer)

        toy_data = {
            "donor_id": current_user.user_id,
            "toy_type": toy_type[idx],
            "is_donatable": donation_status,
            "description": description[idx],
            "image_url": image_path,
            "donor_status": DonorStatus.PENDING,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        toy_stock_crud.create_toy(db, toy_data)
        success_count += 1

    # 성공적으로 등록된 물품 수만큼 포인트 추가
    current_user.points += 100 * success_count
    db.commit()

    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{success_count}개 등록, 오류 {len(errors)}개, 포인트 {100 * success_count}점 적립",
                "errors": errors
            }
        )

    return {
        "success": True,
        "message": f"{success_count}개의 기부물품이 등록되었습니다. 포인트 {100 * success_count}점 적립되었습니다.",
        "points_added": 100 * success_count,
        "current_points": current_user.points
    }


@router.get("/donation/donation_list", response_model=list[toy_stock_schema.DonationList])
async def get_donation_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return toy_stock_crud.get_donation_history(db, current_user.user_id)

