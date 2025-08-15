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

def create_toy(db, toy_data):
    toy = Toy_Stock(**toy_data)
    db.add(toy)
    db.commit()
    db.refresh(toy)
    return toy

def create_toys_bulk(db, toys_data_list):
    """여러 장난감을 일괄 생성하는 함수 (트랜잭션 안정성 보장)"""
    toys = []
    for toy_data in toys_data_list:
        toy = Toy_Stock(**toy_data)
        db.add(toy)
        toys.append(toy)
    db.commit()  # 모든 장난감을 한 번에 commit
    return toys

def get_sale_history(db: Session, user_id: int):
    try:
        toystocks = db.query(Toy_Stock).filter(Toy_Stock.user_id == user_id).all()  # donor_id에서 user_id로 변경
        result = []
        for toy in toystocks:
            # image_url 데이터 타입 통일 (문자열이면 리스트로 변환)
            image_url = toy.image_url
            if isinstance(image_url, str):
                image_url = [image_url] if image_url else None
            elif image_url is None:
                image_url = None
            # 이미 리스트인 경우는 그대로 사용
            
            result.append({
                "toy_id": toy.toy_id,
                "toy_name": None,  # toy_name 필드가 모델에 없으므로 None으로 설정
                "toy_type": toy.toy_type,
                "image_url": image_url,
                "reject_reason": toy.reject_reason,
                "toy_status": toy.toy_status.value if toy.toy_status else None,  # donor_status에서 toy_status로 변경
                "created_at": toy.created_at,
                "purchase_price": toy.purchase_price
            })
        return result
    except Exception as e:
        print(f"Error in get_sale_history CRUD: {str(e)}")
        raise e

def save_images_with_toy_id(db: Session, image_files, toy_id: int):
    """여러 이미지를 toy_id_순서.확장자 형태로 저장하는 함수"""
    import os
    import shutil
    
    image_paths = []
    
    for idx, image_file in enumerate(image_files):
        # 파일 확장자 추출
        file_extension = os.path.splitext(image_file.filename)[1].lower()
        
        # 새로운 파일명 생성: toy_id_순서.확장자
        new_filename = f"{toy_id}_{idx + 1}{file_extension}"
        
        # 저장 경로
        save_dir = "toypics"
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, new_filename)
        
        # 이미지 저장
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        image_paths.append(image_path)
    
    return image_paths