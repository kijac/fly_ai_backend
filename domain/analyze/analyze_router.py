from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
import tempfile
import os
import json
from .analyze_schema import AnalyzeResult

router = APIRouter()

# AI 모델 통합 분석 함수
def run_ai_analysis(
    front_image_bytes: bytes,
    left_image_bytes: Optional[bytes] = None,
    rear_image_bytes: Optional[bytes] = None,
    right_image_bytes: Optional[bytes] = None
) -> dict:
    """
    AI 모델을 사용하여 장난감 분석을 수행하고 완전한 JSON 결과를 반환
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
            from ai_agent_output_json import run_full_pipeline
            
            result = run_full_pipeline(
                query_image_path=front_temp.name,
                left=left_temp.name if left_temp else None,
                rear=rear_temp.name if rear_temp else None,
                right=right_temp.name if right_temp else None,
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

@router.post("/analyze", response_model=dict)
async def analyze_images(
    images: List[UploadFile] = File(...)  # 연속으로 받는 이미지들
):
    """
    장난감 이미지를 AI 모델로 분석하여 완전한 JSON 결과를 반환
    
    - images: 연속으로 업로드된 이미지들 (첫 번째가 메인 이미지)
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
        
        # AI 모델 분석 실행
        result = run_ai_analysis(
            front_image_bytes=front_bytes,
            left_image_bytes=left_bytes,
            rear_image_bytes=rear_bytes,
            right_image_bytes=right_bytes
        )
        
        # 완전한 JSON 결과 반환
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

# 기존 함수는 호환성을 위해 유지 (선택사항)
def convert_result_keys(result):
    return {
        "toy_type": result.get("장난감 종류"),
        "battery": result.get("건전지 여부"),
        "material": result.get("재료"),
        "damage": result.get("파손"),
        "donate": result.get("기부 가능 여부") == "가능",
        "donate_reason": result.get("기부 불가 사유") or "",
        "repair_or_disassemble": result.get("수리/분해"),
        "token_usage": result.get("토큰_사용량"),
    }

@router.post("/analyze_legacy", response_model=list[AnalyzeResult])
async def analyze_images_legacy(photos: list[UploadFile] = File(...)):
    """
    기존 호환성을 위한 레거시 분석 API
    """
    supervisor = SupervisorAgent()
    results = []
    for file in photos:
        image_bytes = await file.read()
        result = supervisor.process(image_bytes)
        results.append(convert_result_keys(result))
    return results