from sqlalchemy import or_
from sqlalchemy.orm import Session

from model import Toy_Stock

def get_toystock_list(db: Session, toy_type: str ='', skip: int = 0, limit: int = 10, keyword: str = ''):
    _toystock_list = db.query(Toy_Stock)

    # 장난감 종류 검색
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
    toy = db.query(Toy_Stock).get(toy_id)
    return toy

from model import Toy_Stock

def create_toy(db, toy_data):
    toy = Toy_Stock(**toy_data)
    db.add(toy)
    db.commit()
    db.refresh(toy)
    return toy