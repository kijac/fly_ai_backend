# toy_stock 테이블 초기화 및 예시 데이터 생성
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from sqlalchemy.orm import Session
from database import engine, get_db
from model import Toy_Stock, ToyStatus, DonationStatus
from datetime import datetime

# 1. toy_stock 테이블의 모든 데이터 삭제
def clear_toy_stock_table():
    with Session(engine) as db:
        try:
            # 모든 toy_stock 데이터 삭제
            db.query(Toy_Stock).delete()
            db.commit()
            print("✅ toy_stock 테이블의 모든 데이터가 삭제되었습니다.")
        except Exception as e:
            db.rollback()
            print(f"❌ 테이블 삭제 중 오류: {str(e)}")

# 2. user_id 2번의 예시 데이터 생성
def create_sample_data():
    with Session(engine) as db:
        try:
            # 예시 데이터 리스트
            sample_toys = [
                {
                    "user_id": 2,
                    "toy_type": "레고",
                    "description": "레고 시리즈 완성품입니다. 상태 좋습니다.",
                    "image_url": ["toypics/sample_1.jpg", "toypics/sample_1_2.jpg"],  # JSON 배열로 저장
                    "toy_status": ToyStatus.FOR_SALE,
                    "sale_price": 15000,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                },
                {
                    "user_id": 2,
                    "toy_type": "인형",
                    "description": "귀여운 곰인형입니다. 깨끗합니다.",
                    "image_url": ["toypics/sample_2.png"],  # JSON 배열로 저장
                    "toy_status": ToyStatus.FOR_SALE,
                    "sale_price": 8000,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                },
                {
                    "user_id": 2,
                    "toy_type": "자동차",
                    "description": "리모컨 자동차입니다. 배터리 포함.",
                    "image_url": ["toypics/sample_3.jpg", "toypics/sample_3_2.jpg", "toypics/sample_3_3.jpg"],  # JSON 배열로 저장
                    "toy_status": ToyStatus.FOR_SALE,
                    "sale_price": 25000,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                },
                {
                    "user_id": 2,
                    "toy_type": "퍼즐",
                    "description": "1000피스 퍼즐입니다. 완성된 상태.",
                    "image_url": ["toypics/sample_4.png", "toypics/sample_4_2.png"],  # JSON 배열로 저장
                    "toy_status": ToyStatus.FOR_SALE,
                    "sale_price": 5000,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                },
                {
                    "user_id": 2,
                    "toy_type": "기타",
                    "description": "기타 장난감들입니다.",
                    "image_url": ["toypics/sample_5.jpg"],  # JSON 배열로 저장
                    "toy_status": ToyStatus.FOR_SALE,
                    "sale_price": 0,  # 무료
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            ]
            
            # 데이터베이스에 삽입
            for toy_data in sample_toys:
                toy = Toy_Stock(**toy_data)
                db.add(toy)
            
            db.commit()
            print(f"✅ user_id 2번의 예시 데이터 {len(sample_toys)}개가 생성되었습니다.")
            
            # 생성된 데이터 확인
            created_toys = db.query(Toy_Stock).filter(Toy_Stock.user_id == 2).all()
            print(f"📊 생성된 장난감 목록:")
            for toy in created_toys:
                print(f"  - ID: {toy.toy_id}, 타입: {toy.toy_type}, 가격: {toy.sale_price}원")
                
        except Exception as e:
            db.rollback()
            print(f"❌ 예시 데이터 생성 중 오류: {str(e)}")

# 3. 실행
if __name__ == "__main__":
    print("🧹 toy_stock 테이블 초기화 및 예시 데이터 생성 시작...")
    
    # 테이블 초기화
    clear_toy_stock_table()
    
    # 예시 데이터 생성
    create_sample_data()
    
    print("🎉 완료!")
