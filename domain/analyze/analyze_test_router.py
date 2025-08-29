from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
import tempfile
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from .analyze_schema import AnalyzeResult

router = APIRouter()

# ThreadPoolExecutor 설정 (동시 요청 처리용)
executor = ThreadPoolExecutor(max_workers=4)  # 최대 4개 동시 처리

# AI 모델 통합 분석 함수 (기존과 동일)
def run_ai_analysis(
    front_image_bytes: bytes,
    left_image_bytes: Optional[bytes] = None,
    rear_image_bytes: Optional[bytes] = None,
    right_image_bytes: Optional[bytes] = None
) -> dict:
    """
    새로운 AI 모델을 사용하여 장난감 분석을 수행하고 완전한 JSON 결과를 반환
    """
    try:
        # 임시 파일로 저장 (AI 모델이 파일 경로를 요구하므로)
        temp_files = []
        
        # 메인 이미지 (필수)
        front_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        front_temp.write(front_image_bytes)
        front_temp.close()
        temp_files.append(front_temp.name)
        
        # 추가 이미지들 (선택사항)
        left_temp = None
        rear_temp = None
        right_temp = None
        
        if left_image_bytes:
            left_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            left_temp.write(left_image_bytes)
            left_temp.close()
            temp_files.append(left_temp.name)
            
        if rear_image_bytes:
            rear_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            rear_temp.write(rear_image_bytes)
            rear_temp.close()
            temp_files.append(rear_temp.name)
            
        if right_image_bytes:
            right_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            right_temp.write(right_image_bytes)
            right_temp.close()
            temp_files.append(right_temp.name)
        
        # AI 모델 분석 실행
        import sys
        ai_agent_path = os.path.join(os.path.dirname(__file__), 'ai_agent')
        sys.path.append(ai_agent_path)
        
        # 작업 디렉토리를 AI 모델 폴더로 변경
        original_cwd = os.getcwd()
        os.chdir(ai_agent_path)
        
        try:
            from ai_agent_output import run_full_pipeline
            
            result = run_full_pipeline(
                used_path=front_temp.name,
                base_bias=0.7
            )
        finally:
            # 작업 디렉토리 복원
            os.chdir(original_cwd)
        
        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return result
        
    except Exception as e:
        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류 발생: {str(e)}")

@router.post("/analyze_test", response_model=dict)
async def analyze_images_test(
    images: List[UploadFile] = File(...)  # 연속으로 받는 이미지들
):
    """
    장난감 이미지를 새로운 AI 모델로 분석하여 완전한 JSON 결과를 반환 (ThreadPoolExecutor 사용)
    
    - images: 연속으로 업로드된 이미지들 (첫 번째가 메인 이미지)
    - 반환: toy_name, retail_price, purchase_price, soil, damage, toy_type, material
    - 모든 요청: ThreadPoolExecutor (병렬 처리)
    """
    try:
        if not images:
            raise HTTPException(status_code=400, detail="이미지가 필요합니다.")
        
        # 이미지 파일 읽기
        image_bytes_list = []
        for image in images:
            image_bytes = await image.read()
            image_bytes_list.append(image_bytes)
        
        # 첫 번째 이미지는 메인 이미지 (필수)
        front_bytes = image_bytes_list[0]
        
        # 나머지 이미지들은 순서대로 left, rear, right로 할당
        left_bytes = image_bytes_list[1] if len(image_bytes_list) > 1 else None
        rear_bytes = image_bytes_list[2] if len(image_bytes_list) > 2 else None
        right_bytes = image_bytes_list[3] if len(image_bytes_list) > 3 else None
        
        # 모든 요청을 ThreadPoolExecutor로 처리 (병렬 처리)
        print(f"⚡ ThreadPoolExecutor processing: {len(image_bytes_list)} images")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            run_ai_analysis,
            front_bytes,
            left_bytes,
            rear_bytes,
            right_bytes
        )
        
        # 완전한 JSON 결과 반환
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

@router.post("/analyze_test_batch", response_model=List[dict])
async def analyze_images_test_batch(
    image_batches: List[List[UploadFile]] = File(...)  # 여러 이미지 배치
):
    """
    여러 이미지 배치를 동시에 분석 (ThreadPoolExecutor 사용)
    
    - image_batches: 여러 이미지 배치의 리스트
    - 각 배치는 이미지들의 리스트
    - ThreadPoolExecutor를 사용하여 배치별로 동시 처리
    """
    try:
        if not image_batches:
            raise HTTPException(status_code=400, detail="이미지 배치가 필요합니다.")
        
        # 각 배치의 이미지들을 bytes로 변환
        batch_data = []
        for batch in image_batches:
            batch_bytes = []
            for image in batch:
                image_bytes = await image.read()
                batch_bytes.append(image_bytes)
            batch_data.append(batch_bytes)
        
        # ThreadPoolExecutor를 사용하여 모든 배치를 동시에 처리
        loop = asyncio.get_event_loop()
        
        # 각 배치를 executor에 제출
        futures = []
        for batch_bytes in batch_data:
            if batch_bytes:  # 빈 배치가 아닌 경우만
                front_bytes = batch_bytes[0]
                left_bytes = batch_bytes[1] if len(batch_bytes) > 1 else None
                rear_bytes = batch_bytes[2] if len(batch_bytes) > 2 else None
                right_bytes = batch_bytes[3] if len(batch_bytes) > 3 else None
                
                future = loop.run_in_executor(
                    executor,
                    run_ai_analysis,
                    front_bytes,
                    left_bytes,
                    rear_bytes,
                    right_bytes
                )
                futures.append(future)
        
        # 모든 결과를 동시에 수집
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 예외가 발생한 결과는 에러 메시지로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": f"배치 {i+1} 처리 중 오류: {str(result)}",
                    "batch_index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 분석 중 오류 발생: {str(e)}")

# 기존 호환성을 위한 레거시 함수들 (ThreadPoolExecutor 적용)
def convert_result_keys(result):
    return {
        "toy_type": result.get("toy_type"),
        "battery": result.get("건전지"),
        "material": result.get("material"),
        "damage": result.get("damage"),
        "donate": result.get("donate", False),
        "donate_reason": result.get("donate_reason", ""),
        "repair_or_disassemble": result.get("repair_or_disassemble", ""),
        "token_usage": result.get("에이전트_토큰합", 0),
    }

@router.post("/analyze_test_legacy", response_model=List[AnalyzeResult])
async def analyze_images_test_legacy(photos: list[UploadFile] = File(...)):
    """
    기존 호환성을 위한 레거시 분석 API (ThreadPoolExecutor 사용)
    """
    try:
        # 이미지 파일들을 bytes로 변환
        image_bytes_list = []
        for file in photos:
            image_bytes = await file.read()
            image_bytes_list.append(image_bytes)
        
        # ThreadPoolExecutor를 사용하여 모든 이미지를 동시에 처리
        loop = asyncio.get_event_loop()
        
        # 각 이미지를 executor에 제출
        futures = []
        for image_bytes in image_bytes_list:
            future = loop.run_in_executor(
                executor,
                run_ai_analysis,
                image_bytes
            )
            futures.append(future)
        
        # 모든 결과를 동시에 수집
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 예외가 발생한 결과는 기본값으로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(convert_result_keys({
                    "toy_type": "unknown",
                    "건전지": "unknown",
                    "material": "unknown",
                    "damage": "unknown",
                    "donate": False,
                    "donate_reason": f"오류: {str(result)}",
                    "repair_or_disassemble": "unknown",
                    "에이전트_토큰합": 0
                }))
            else:
                processed_results.append(convert_result_keys(result))
        
        return processed_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"레거시 분석 중 오류 발생: {str(e)}")

# ThreadPoolExecutor 상태 확인용 엔드포인트
@router.get("/analyze_test/status")
async def get_executor_status():
    """
    ThreadPoolExecutor 상태 확인
    """
    return {
        "max_workers": executor._max_workers,
        "active_threads": len(executor._threads),
        "queue_size": executor._work_queue.qsize() if hasattr(executor, '_work_queue') else 0
    }
