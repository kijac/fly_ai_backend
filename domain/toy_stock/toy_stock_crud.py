from sqlalchemy import or_
from sqlalchemy.orm import Session
from typing import Tuple, List
import json
from model import Toy_Stock
from fastapi import HTTPException, status

def get_toystock_list(db: Session, toy_type: str ='', skip: int = 0, limit: int = 10, keyword: str = '', toy_status: str = 'for_sale')-> Tuple[int, List[Toy_Stock]]:
    
    _toystock_list = db.query(Toy_Stock)

    # 상태 필터
    if toy_status:
        _toystock_list = _toystock_list.filter(Toy_Stock.toy_status == toy_status)

    # 장난감 종류 필터
    if toy_type:
        search = '%%{}%%'.format(toy_type)
        _toystock_list = _toystock_list.filter(
            Toy_Stock.toy_type.ilike(search) # 장난감 종류만 따로 검색
        )

    # 키워드 검색
    if keyword:
        search = '%%{}%%'.format(keyword)
        _toystock_list = _toystock_list.filter(\
            or_(
                Toy_Stock.toy_name.ilike(search), # 장난감 이름
                Toy_Stock.toy_type.ilike(search), # 장난감 종류
                Toy_Stock.description.ilike(search) # 장난감 설명
            )
        )

    # 전체 개수 (정렬 없이 계산)
    total = _toystock_list.count()

    # 정렬 + 페이지네이션
    _toystock_list = (
        _toystock_list.order_by(Toy_Stock.updated_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    )
    return total, _toystock_list

def get_toy(db: Session, toy_id: int):
    toy = db.get(Toy_Stock, toy_id)

    if toy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="장난감을 찾을 수 없습니다.")

    if isinstance(toy.image_url, str):
        try:
            toy.image_url = json.loads(toy.image_url)
        except Exception:
            toy.image_url = None   
    return toy