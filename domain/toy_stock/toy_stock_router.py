from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from starlette import status
from typing import List, Optional

from database import get_db
from domain.toy_stock import toy_stock_schema, toy_stock_crud

import os
import shutil
from fastapi import Form, File, UploadFile
from model import DonationStatus

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

from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from sqlalchemy.orm import Session
from datetime import datetime
import os, shutil

from database import get_db
from domain.toy_stock import toy_stock_schema, toy_stock_crud
from model import DonationStatus, DonorStatus, User
from domain.user.user_router import get_current_user

router = APIRouter()

@router.post("/donation")
async def register_toy(
    toy_type: str = Form(...),
    is_donatable: str = Form(...),  # ENUM 문자열로 받음
    description: str = Form(""),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 이미지 저장
    save_dir = "toypics"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # ENUM 값 변환
    try:
        donation_status = DonationStatus(is_donatable)
    except ValueError:
        raise HTTPException(status_code=400, detail="is_donatable 값이 올바르지 않습니다.")

    # toy_stock 테이블에 저장할 데이터 준비
    toy_data = {
        "donor_id": current_user.user_id,           # 로그인 사용자 정보
        "toy_type": toy_type,
        "is_donatable": donation_status,
        "description": description,
        "image_url": image_path,
        "donor_status": DonorStatus.PENDING,        # 기본값
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        # toy_id는 자동생성
    }
    toy = toy_stock_crud.create_toy(db, toy_data)
    return {
        "success": True,
        "message": "기부물품 등록이 완료되었습니다.",
    }