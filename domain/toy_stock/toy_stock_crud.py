from sqlalchemy import or_
from sqlalchemy.orm import Session

from model import Toy_Stock

def get_toystock_list(db: Session, skip: int = 0, limit: int = 10, keyword: str = ''):
    _toystock_list = db.query(Toy_Stock)

    # 키워드 검색
    if keyword:
        search = '%%{}%%'.format(keyword)
        _toystock_list = _toystock_list.filter(\
            or_(
                Toy_Stock.toy_name.ilike(search),
                Toy_Stock.toy_type.ilike(search),
                Toy_Stock.description.ilike(search)
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
