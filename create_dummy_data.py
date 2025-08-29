#!python
"""
Toy Stock 더미 데이터 생성 스크립트

CSV 파일을 읽어서 toy_stock 테이블에 더미 데이터를 삽입합니다.
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append('.')

# 데이터베이스 모델 import
from database import SessionLocal, engine
from model import Toy_Stock, User, ToyStatus

def read_csv_file(csv_path):
    """CSV 파일을 읽는 함수"""
    try:
        # UTF-8로 먼저 시도
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # CP949로 시도
            df = pd.read_csv(csv_path, encoding='cp949')
        except UnicodeDecodeError:
            # UTF-8-sig로 시도
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print(f"📊 CSV 파일 읽기 완료: {len(df)} 행")
    print(f"📋 원본 컬럼: {list(df.columns)}")
    
    return df

def preprocess_csv_data(df):
    """CSV 데이터 전처리"""
    # 첫 번째 행이 헤더인지 확인하고 처리
    if '이미지 (전, 좌, 후, 우)' in df.columns or 'Unnamed: 1' in df.columns:
        # 첫 번째 행이 헤더인 경우, 첫 번째 행을 건너뛰고 데이터만 사용
        df = df.iloc[1:].reset_index(drop=True)
        print("🔧 첫 번째 행(헤더) 제거 완료")
    
    # 컬럼명 정리
    print(f"🔍 실제 컬럼 개수: {len(df.columns)}")
    print(f"🔍 실제 컬럼명: {list(df.columns)}")
    
    if len(df.columns) == 8:
        # 8개 컬럼인 경우 (IMG_NO 포함)
        df.columns = ['img_no', 'front_img', 'left_img', 'rear_img', 'right_img', 'toy_name', 'toy_type', 'material']
        # img_no 컬럼 제거
        df = df.drop('img_no', axis=1)
    elif len(df.columns) == 7:
        # 7개 컬럼인 경우
        df.columns = ['front_img', 'left_img', 'rear_img', 'right_img', 'toy_name', 'toy_type', 'material']
    else:
        print(f"⚠️ 예상치 못한 컬럼 개수: {len(df.columns)}")
        # 동적으로 컬럼명 할당
        expected_columns = ['front_img', 'left_img', 'rear_img', 'right_img', 'toy_name', 'toy_type', 'material']
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns + [f'extra_{i}' for i in range(len(df.columns) - len(expected_columns))]
        else:
            print(f"❌ 컬럼이 부족합니다. 최소 {len(expected_columns)}개 필요")
            return df
    
    print("🔧 컬럼명 정리 완료")
    print(f"📋 새로운 컬럼: {list(df.columns)}")
    print(f"📊 처리된 데이터 행 수: {len(df)}")
    
    return df

def check_image_files(df, toypics_dir):
    """이미지 파일 존재 여부 확인"""
    image_files = os.listdir(toypics_dir) if os.path.exists(toypics_dir) else []
    
    print(f"📁 toypics 폴더 이미지 파일 수: {len(image_files)}")
    print(f"🖼️ 샘플 이미지 파일들: {image_files[:5]}")
    
    def check_image_exists(img_name):
        if pd.isna(img_name):
            return False
        
        # 확장자 변환 시도 (.JPG -> .jpg, .png -> .png)
        base_name = str(img_name).strip()
        
        # 원본 이름으로 확인
        if base_name in image_files:
            return True
        
        # 확장자 변환해서 확인
        if base_name.endswith('.JPG'):
            jpg_name = base_name.replace('.JPG', '.jpg')
            if jpg_name in image_files:
                return True
        
        # 공백 제거 후 확인 (실제 파일명에 공백이 있을 수 있음)
        for file_name in image_files:
            if file_name.strip() == base_name:
                return True
            # 확장자 제거 후 비교
            if file_name.strip().split('.')[0] == base_name.split('.')[0]:
                return True
        
        return False
    
    # 각 행의 이미지 파일 존재 여부 확인
    df['front_exists'] = df['front_img'].apply(check_image_exists)
    df['left_exists'] = df['left_img'].apply(check_image_exists)
    df['rear_exists'] = df['rear_img'].apply(check_image_exists)
    df['right_exists'] = df['right_img'].apply(check_image_exists)
    
    print(f"✅ 모든 이미지가 존재하는 행: {(df['front_exists'] & df['left_exists'] & df['rear_exists'] & df['right_exists']).sum()}")
    print(f"❌ 일부 이미지가 없는 행: {len(df) - (df['front_exists'] & df['left_exists'] & df['rear_exists'] & df['right_exists']).sum()}")
    
    # 디버깅: 매칭되지 않는 파일들 확인
    print("\n🔍 매칭되지 않는 파일들:")
    for idx, row in df.iterrows():
        if not (row['front_exists'] and row['left_exists'] and row['rear_exists'] and row['right_exists']):
            print(f"  행 {idx}: {row['toy_name']}")
            if not row['front_exists']:
                print(f"    front: {row['front_img']} (찾을 수 없음)")
            if not row['left_exists']:
                print(f"    left: {row['left_img']} (찾을 수 없음)")
            if not row['rear_exists']:
                print(f"    rear: {row['rear_img']} (찾을 수 없음)")
            if not row['right_exists']:
                print(f"    right: {row['right_img']} (찾을 수 없음)")
    
    return df

def get_or_create_user():
    """사용자 조회 또는 생성"""
    db = SessionLocal()
    
    try:
        # 기존 사용자 확인
        existing_users = db.query(User).all()
        print(f"👥 기존 사용자 수: {len(existing_users)}")
        
        if not existing_users:
            # 더미 사용자 생성
            dummy_user = User(
                name="테스트 사용자",
                email="test@example.com",
                password="hashed_password",
                address="서울시 강남구",
                role="USER",
                phone_number="010-1234-5678",
                created_at=datetime.now(),
                points=100,
                subscribe=False
            )
            
            db.add(dummy_user)
            db.commit()
            db.refresh(dummy_user)
            
            print(f"✅ 더미 사용자 생성 완료: ID {dummy_user.user_id}")
            user_id = dummy_user.user_id
        else:
            user_id = existing_users[0].user_id
            print(f"✅ 기존 사용자 사용: ID {user_id}")
            
    except Exception as e:
        print(f"❌ 사용자 생성/조회 실패: {e}")
        db.rollback()
        user_id = 1  # 기본값
    finally:
        db.close()
    
    return user_id

def create_image_url_json(front_img, left_img, rear_img, right_img):
    """이미지 파일명들을 JSON 형태로 변환"""
    images = []
    
    if not pd.isna(front_img):
        images.append(f"toypics/{front_img.strip()}")
    if not pd.isna(left_img):
        images.append(f"toypics/{left_img.strip()}")
    if not pd.isna(rear_img):
        images.append(f"toypics/{rear_img.strip()}")
    if not pd.isna(right_img):
        images.append(f"toypics/{right_img.strip()}")
    
    return json.dumps(images, ensure_ascii=False)

def generate_random_price():
    """30000~40000 사이의 랜덤 가격 생성"""
    return random.randint(30000, 40000)

def generate_random_date():
    """최근 30일 내의 랜덤 날짜 생성"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    
    return random_date

def create_toy_stock_data(df, user_id):
    """Toy Stock 데이터 생성"""
    toy_stock_data = []
    
    for idx, row in df.iterrows():
        # 모든 이미지가 존재하는 경우만 처리
        if row['front_exists'] and row['left_exists'] and row['rear_exists'] and row['right_exists']:
            
            # 이미지 URL JSON 생성
            image_url_json = create_image_url_json(
                row['front_img'], 
                row['left_img'], 
                row['rear_img'], 
                row['right_img']
            )
            
            # 랜덤 가격 생성
            sale_price = generate_random_price()
            
            # 랜덤 날짜 생성
            created_date = generate_random_date()
            
            toy_data = {
                'user_id': user_id,
                'toy_name': str(row['toy_name']).strip() if not pd.isna(row['toy_name']) else f"장난감_{idx+1}",
                'toy_type': str(row['toy_type']).strip() if not pd.isna(row['toy_type']) else "기타",
                'image_url': image_url_json,
                'toy_status': ToyStatus.FOR_SALE,
                'sale_price': sale_price,
                'purchase_price': int(sale_price * 0.7),  # 매입가는 판매가의 70%
                'material': str(row['material']).strip() if not pd.isna(row['material']) else "기타",
                'created_at': created_date,
                'updated_at': created_date,
                'description': f"{row['toy_name']} - {row['toy_type']} 종류의 장난감입니다." if not pd.isna(row['toy_name']) else f"장난감_{idx+1} - 기타 종류의 장난감입니다."
            }
            
            toy_stock_data.append(toy_data)
    
    print(f"📦 생성된 Toy Stock 데이터: {len(toy_stock_data)}개")
    print("\n📋 샘플 데이터:")
    for i, data in enumerate(toy_stock_data[:3]):
        print(f"  {i+1}. {data['toy_name']} - {data['toy_type']} - {data['sale_price']:,}원")
    
    return toy_stock_data

def insert_to_database(toy_stock_data):
    """데이터베이스에 삽입"""
    db = SessionLocal()
    inserted_count = 0
    
    try:
        for toy_data in toy_stock_data:
            # Toy_Stock 객체 생성
            toy_stock = Toy_Stock(**toy_data)
            
            # 데이터베이스에 추가
            db.add(toy_stock)
            inserted_count += 1
        
        # 커밋
        db.commit()
        print(f"✅ 성공적으로 {inserted_count}개의 Toy Stock 데이터가 삽입되었습니다!")
        
    except Exception as e:
        print(f"❌ 데이터 삽입 실패: {e}")
        db.rollback()
    finally:
        db.close()
    
    return inserted_count

def verify_inserted_data():
    """삽입된 데이터 확인"""
    db = SessionLocal()
    
    try:
        # 전체 Toy Stock 데이터 조회
        all_toys = db.query(Toy_Stock).all()
        print(f"📊 전체 Toy Stock 데이터 수: {len(all_toys)}")
        
        # 상태별 통계
        status_counts = db.query(Toy_Stock.toy_status, db.func.count(Toy_Stock.toy_id)).\
            group_by(Toy_Stock.toy_status).all()
        
        print("\n📈 상태별 통계:")
        for status, count in status_counts:
            print(f"  {status}: {count}개")
        
        # 샘플 데이터 출력
        print("\n📋 샘플 데이터:")
        sample_toys = db.query(Toy_Stock).limit(5).all()
        for toy in sample_toys:
            print(f"  ID: {toy.toy_id}, 이름: {toy.toy_name}, 종류: {toy.toy_type}, 가격: {toy.sale_price:,}원")
            
    except Exception as e:
        print(f"❌ 데이터 조회 실패: {e}")
    finally:
        db.close()

def main():
    """메인 함수"""
    print("🚀 Toy Stock 더미 데이터 생성 시작...")
    
    # 1. CSV 파일 읽기
    csv_path = 'datasets.CSV'
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    df = read_csv_file(csv_path)
    
    # 2. CSV 데이터 전처리
    df = preprocess_csv_data(df)
    
    # 3. 이미지 파일 존재 확인
    toypics_dir = 'toypics'
    df = check_image_files(df, toypics_dir)
    
    # 4. 사용자 조회 또는 생성
    user_id = get_or_create_user()
    
    # 5. Toy Stock 데이터 생성
    toy_stock_data = create_toy_stock_data(df, user_id)
    
    if not toy_stock_data:
        print("❌ 삽입할 데이터가 없습니다.")
        return
    
    # 6. 데이터베이스에 삽입
    inserted_count = insert_to_database(toy_stock_data)
    
    if inserted_count > 0:
        # 7. 삽입된 데이터 확인
        verify_inserted_data()
        
        # 8. 완료 메시지
        print("\n🎉 더미 데이터 생성이 완료되었습니다!")
        print("\n📝 생성된 데이터 특징:")
        print("  • toy_name: CSV의 공식 제품명")
        print("  • toy_type: CSV의 장난감 종류")
        print("  • image_url: toypics 폴더의 상대경로 (JSON 형태)")
        print("  • sale_price: 30,000~40,000원 랜덤")
        print("  • toy_status: FOR_SALE")
        print("  • purchase_price: sale_price의 70%")
        print("  • created_at: 최근 30일 내 랜덤 날짜")
        print("\n🔍 확인 방법:")
        print("  • MySQL Workbench에서 'SELECT * FROM toy_stock;' 실행")
        print("  • FastAPI 서버 실행 후 API 엔드포인트 테스트")

if __name__ == "__main__":
    main()
