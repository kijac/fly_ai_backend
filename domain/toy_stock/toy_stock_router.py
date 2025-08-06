from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from starlette import status
from typing import List, Optional

from database import get_db
from domain.toy_stock import toy_stock_schema, toy_stock_crud

router = APIRouter(
    prefix="/api/toy_stock",
)

# 장난감 리스트 불러오기
@router.get("/toystokclist", response_model=toy_stock_schema.ToyStockList)
def toystock_list(db: Session = Depends(get_db),
    page: int = Query(0, ge=0, description="페이지 번호 (0부터 시작)"),
    size: int = Query(10, ge=1, le=50, description="한 페이지에 보여줄 개수"),
    keyword: str = Query("", description="검색 키워드 (없으면 전체 조회)")): # le: less than or equal to, 1이상 50이하
    ##
    total, _toystock_list = toy_stock_crud.get_toystock_list(db, skip=page*size, limit=size, keyword=keyword)
    return {
        "total": total, 
        "toystock_list": _toystock_list
    }

# 장난감 Detail 정보 불러오기
#@router.get("detail/{toy_id}", status_code=status.HTTP_200_OK)


## 주문한 장난감 리스트 불러오기
#@router.get("/mytoy", status_code=status.HTTP_200_OK)