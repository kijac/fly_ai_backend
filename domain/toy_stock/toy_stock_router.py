from fastapi import APIRouter, Depends, HTTPException, Query, Form, File, UploadFile
from sqlalchemy.orm import Session
from starlette import status
from typing import List, Optional
import os
import shutil

from database import get_db
from domain.toy_stock import toy_stock_schema, toy_stock_crud
from fastapi import Form, File, UploadFile
from model import ToyStatus, User, DonationStatus
from datetime import datetime
from domain.user.user_router import get_current_user

router = APIRouter(
    prefix="/api/toy_stock",
)

# 장난감 리스트 불러오기
@router.get("/toystocklist", response_model=toy_stock_schema.ToyStockList)
def toystock_list(
    toy_type: str = Query("", description="장난감 종류 검색 (없으면 전체 조회)"), # Path Parameter
    page: int = Query(0, ge=0, description="페이지 번호 (0부터 시작)"),
    size: int = Query(10, ge=1, le=50, description="한 페이지에 보여줄 개수"),
    keyword: str = Query("", description="검색 키워드 (없으면 전체 조회)"),
    toy_status: str = Query("for_sale", description="장난감 상태 검색 (없으면 'for sale'로 조회)"),
    db: Session = Depends(get_db)
    ):
    total, _toystock_list = toy_stock_crud.get_toystock_list(db, toy_type=toy_type, skip=page*size, limit=size, keyword=keyword, toy_status=toy_status)
    return {
        "total": total, 
        "toystock_list": _toystock_list
    }

# 테스트용: 이미지 URL 포함 장난감 리스트
@router.get("/toystocklist_remote", response_model=toy_stock_schema.ToyStockList)
def toystock_list_test(
    toy_type: str = Query("", description="장난감 종류 검색 (없으면 전체 조회)"),
    page: int = Query(0, ge=0, description="페이지 번호 (0부터 시작)"),
    size: int = Query(10, ge=1, le=50, description="한 페이지에 보여줄 개수"),
    keyword: str = Query("", description="검색 키워드 (없으면 전체 조회)"),
    toy_status: str = Query("for_sale", description="장난감 상태 검색 (없으면 'for sale'로 조회)"),
    db: Session = Depends(get_db)
    ):
    """
    원격 프론트엔드 접속용
    """
    total, _toystock_list = toy_stock_crud.get_toystock_list(db, toy_type=toy_type, skip=page*size, limit=size, keyword=keyword, toy_status=toy_status)
    
    # 테스트용: 이미지 URL 변환
    for toy in _toystock_list:
        if isinstance(toy.image_url, dict):
            # 기존 딕셔너리를 URL로 변환
            url_dict = {}
            for key, value in toy.image_url.items():
                if value:
                    url_dict[key] = f"/api/toy_stock/images/{value}"
                else:
                    url_dict[key] = None
            toy.image_url = url_dict
        elif isinstance(toy.image_url, list):
            # 리스트를 URL 딕셔너리로 변환
            if toy.image_url:
                url_dict = {"main": f"/api/toy_stock/images/{toy.image_url[0]}"}
                for i, img in enumerate(toy.image_url[1:], 1):
                    url_dict[f"sub{i}"] = f"/api/toy_stock/images/{img}"
                toy.image_url = url_dict
            else:
                toy.image_url = None
        elif isinstance(toy.image_url, str):
            # 문자열을 URL로 변환
            toy.image_url = {"main": f"/api/toy_stock/images/{toy.image_url}"}
        elif toy.image_url is None:
            toy.image_url = None
    
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


@router.post("/submit_sale", response_model=toy_stock_schema.ToySaleResponse)
async def register_toys_bulk(
    toy_name: List[str] = Form(...),
    toy_type: List[str] = Form(...),
    soil: List[str] = Form(...),
    damage: List[str] = Form(...),
    sale_price: List[int] = Form(...),
    images: List[UploadFile] = File(...),  # 모든 이미지를 하나의 리스트로 받음
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not toy_name or not toy_type or not soil or not damage or not sale_price or not images:
        raise HTTPException(status_code=400, detail="필수 항목이 누락되었습니다.")

    if not (len(toy_name) == len(toy_type) == len(soil) == len(damage) == len(sale_price)):
        raise HTTPException(status_code=400, detail="입력 데이터의 개수가 일치하지 않습니다.")
    
    # 이미지 개수로 장난감 개수 자동 계산 (자유로운 개수)
    if len(images) < len(toy_name):
        raise HTTPException(status_code=400, detail=f"이미지 개수가 부족합니다. 장난감 {len(toy_name)}개, 이미지 {len(images)}장")
    
    # 각 장난감별 이미지 개수 계산 (균등 분배)
    base_count = len(images) // len(toy_name)  # 기본 개수
    remainder = len(images) % len(toy_name)    # 나머지
    
    toy_image_counts = []
    for i in range(len(toy_name)):
        if i < remainder:
            toy_image_counts.append(base_count + 1)  # 나머지를 앞쪽 장난감들에게 1개씩 추가
        else:
            toy_image_counts.append(base_count)

    errors = []
    toys_data_list = []
    
    # 1단계: 입력 검증
    for idx in range(len(toy_name)):
        # 판매가 검증 (0원도 허용)
        if sale_price[idx] < 0:
            errors.append(f"{toy_name[idx]}: 판매가는 0 이상이어야 합니다.")
            continue

        # 이미지 파일 확장자 검증 (이미지 인덱스 계산)
        start_idx = sum(toy_image_counts[:idx])  # 이전 장난감들의 이미지 개수 합
        end_idx = start_idx + toy_image_counts[idx]  # 현재 장난감의 이미지 끝 인덱스
        
        for img_idx in range(start_idx, end_idx):
            if not images[img_idx].filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                errors.append(f"{toy_name[idx]} (이미지 {img_idx - start_idx + 1}): 이미지 파일만 업로드 가능합니다.")
                continue

        # 장난감 데이터 준비 (이미지는 나중에 저장)
        toy_data = {
            "user_id": current_user.user_id,
            "toy_name": toy_name[idx],
            "toy_type": toy_type[idx],
            "image_url": None,  # 임시로 None, 나중에 업데이트
            "toy_status": ToyStatus.FOR_SALE,
            "sale_price": sale_price[idx],
            "soil": soil[idx],
            "damage": damage[idx],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        toys_data_list.append(toy_data)

    # 2단계: 트랜잭션으로 일괄 저장
    success_count = 0
    if toys_data_list:
        try:
            # 장난감 데이터 저장
            created_toys = toy_stock_crud.create_toys_bulk(db, toys_data_list)
            success_count = len(toys_data_list)
            
            # 3단계: 이미지를 toy_id 기반으로 저장하고 DB 업데이트
            image_start_idx = 0
            for idx, toy in enumerate(created_toys):
                # 현재 장난감의 이미지들 추출
                image_count = toy_image_counts[idx]
                toy_images = images[image_start_idx:image_start_idx + image_count]
                
                # 여러 이미지를 toy_id_순서.확장자 형태로 저장
                image_paths = toy_stock_crud.save_images_with_toy_id(db, toy_images, toy.toy_id)
                # DB에 이미지 경로들 업데이트 (JSON 배열)
                toy.image_url = image_paths
                
                image_start_idx += image_count
            
            # 성공적으로 등록된 물품 수만큼 포인트 추가 (현재는 0점, 추후 정책에 따라 조정)
            current_user.points += 0 * success_count
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"장난감 등록 중 오류가 발생했습니다: {str(e)}")

    # 3단계: 응답 처리
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{success_count}개 등록, 오류 {len(errors)}개",
                "errors": errors
            }
        )

    return {
        "success": True,
        "message": f"{success_count}개의 장난감이 판매 등록되었습니다.",
        "registered_count": success_count,
        "points_added": 0 * success_count,
        "current_points": current_user.points
    }


@router.post("/submit_donation", response_model=toy_stock_schema.ToyDonationResponse)
async def register_donation_bulk(
    toy_type: List[str] = Form(...),
    soil: List[str] = Form(...),
    damage: List[str] = Form(...),
    is_donatable: List[str] = Form(...),
    images: List[UploadFile] = File(...),  # 모든 이미지를 하나의 리스트로 받음
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not toy_type or not soil or not damage or not is_donatable or not images:
        raise HTTPException(status_code=400, detail="필수 항목이 누락되었습니다.")

    if not (len(toy_type) == len(soil) == len(damage) == len(is_donatable)):
        raise HTTPException(status_code=400, detail="입력 데이터의 개수가 일치하지 않습니다.")
    
    # 이미지 개수로 장난감 개수 자동 계산 (자유로운 개수)
    if len(images) < len(toy_type):
        raise HTTPException(status_code=400, detail=f"이미지 개수가 부족합니다. 장난감 {len(toy_type)}개, 이미지 {len(images)}장")
    
    # 각 장난감별 이미지 개수 계산 (균등 분배)
    base_count = len(images) // len(toy_type)  # 기본 개수
    remainder = len(images) % len(toy_type)    # 나머지
    
    toy_image_counts = []
    for i in range(len(toy_type)):
        if i < remainder:
            toy_image_counts.append(base_count + 1)  # 나머지를 앞쪽 장난감들에게 1개씩 추가
        else:
            toy_image_counts.append(base_count)

    errors = []
    toys_data_list = []
    
    # 1단계: 입력 검증
    for idx in range(len(toy_type)):
        # is_donatable ENUM 값 검증
        try:
            donation_status = DonationStatus(is_donatable[idx])
        except ValueError:
            errors.append(f"{toy_type[idx]}: is_donatable 값이 올바르지 않습니다. (impossible, recyclable, upcycle 중 선택)")
            continue

        # 이미지 파일 확장자 검증 (이미지 인덱스 계산)
        start_idx = sum(toy_image_counts[:idx])  # 이전 장난감들의 이미지 개수 합
        end_idx = start_idx + toy_image_counts[idx]  # 현재 장난감의 이미지 끝 인덱스
        
        for img_idx in range(start_idx, end_idx):
            if not images[img_idx].filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                errors.append(f"{toy_type[idx]} (이미지 {img_idx - start_idx + 1}): 이미지 파일만 업로드 가능합니다.")
                continue

        # 장난감 데이터 준비 (이미지는 나중에 저장)
        toy_data = {
            "user_id": current_user.user_id,
            "toy_type": toy_type[idx],
            "image_url": None,  # 임시로 None, 나중에 업데이트
            "toy_status": ToyStatus.DONATION,  # 기부는 DONATION 상태로 설정
            "is_donatable": donation_status,
            "soil": soil[idx],
            "damage": damage[idx],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        toys_data_list.append(toy_data)

    # 2단계: 트랜잭션으로 일괄 저장
    success_count = 0
    if toys_data_list:
        try:
            # 장난감 데이터 저장
            created_toys = toy_stock_crud.create_toys_bulk(db, toys_data_list)
            success_count = len(toys_data_list)
            
            # 3단계: 이미지를 toy_id 기반으로 저장하고 DB 업데이트
            image_start_idx = 0
            for idx, toy in enumerate(created_toys):
                # 현재 장난감의 이미지들 추출
                image_count = toy_image_counts[idx]
                toy_images = images[image_start_idx:image_start_idx + image_count]
                
                # 여러 이미지를 toy_id_순서.확장자 형태로 저장
                image_paths = toy_stock_crud.save_images_with_toy_id(db, toy_images, toy.toy_id)
                # DB에 이미지 경로들 업데이트 (JSON 배열)
                toy.image_url = image_paths
                
                image_start_idx += image_count
            
            # 성공적으로 등록된 물품 수만큼 포인트 추가 (현재는 0점, 추후 정책에 따라 조정)
            current_user.points += 30 * success_count
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"장난감 등록 중 오류가 발생했습니다: {str(e)}")

    # 3단계: 응답 처리
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{success_count}개 등록, 오류 {len(errors)}개",
                "errors": errors
            }
        )

    return {
        "success": True,
        "message": f"{success_count}개의 장난감이 기부 등록되었습니다. 포인트 {0 * success_count}점 적립되었습니다.",
        "registered_count": success_count,
        "points_added": 0 * success_count,
        "current_points": current_user.points
    }


@router.get("/sale_list", response_model=list[toy_stock_schema.SaleList])
async def get_sale_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        return toy_stock_crud.get_sale_history(db, current_user.user_id)
    except Exception as e:
        print(f"Error in get_sale_history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류가 발생했습니다: {str(e)}")

from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent  # domain/toy_stock의 상위 = domain
MEDIA_DIR = BASE_DIR.parent / "toypics"            # domain의 상위 = 프로젝트 루트의 toypics

# 이미지 직접 서빙 엔드포인트
@router.get("/images/{image_path:path}")
async def serve_image(image_path: str):
    # URL 디코딩 및 공백 제거
    import urllib.parse
    image_path = urllib.parse.unquote(image_path).strip()
    
    # HTTP/1.1 등 HTTP 헤더 정보 제거
    if " HTTP/" in image_path:
        image_path = image_path.split(" HTTP/")[0]
    
    # 추가 URL 정리 (공백, 특수문자 등 제거)
    image_path = image_path.strip()
    
    # 백슬래시 → 슬래시 치환(윈도우 경로 대응)
    p = Path(image_path.replace("\\", "/"))

    # URL에 'toypics/'가 포함돼 들어와도 정상 동작하도록 제거
    parts = list(p.parts)
    if parts and parts[0].lower() == "toypics":
        parts = parts[1:]
    p = Path(*parts)

    # 최종 실제 파일 경로 만들기 (프로젝트 루트 기준)
    target = Path("toypics") / p

    # 디렉터리 탈출 방지
    try:
        target.resolve().relative_to(Path("toypics").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="잘못된 경로 요청입니다.")

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다. 경로: {target}")

    return FileResponse(str(target))

# 테스트용 간단한 이미지 서빙 엔드포인트
@router.get("/test_image/{filename}")
async def test_image(filename: str):
    """
    테스트용: 간단한 이미지 서빙
    """
    import os
    from fastapi.responses import FileResponse
    
    # toypics 폴더의 파일 직접 접근
    image_path = os.path.join("toypics", filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {image_path}")
    
    return FileResponse(image_path)

