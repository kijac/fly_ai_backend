from fastapi import APIRouter, HTTPException, Query, File, UploadFile, Form
from starlette import status
from datetime import datetime
from typing import Optional

from domain.analyze import analyze_schema, analyze_crud

router = APIRouter(
    prefix="/api/analyze",
)

@router.post("/classify", response_model=analyze_schema.AnalyzeResponse)
async def classify_toy(
    image: UploadFile = File(..., description="장난감 이미지 파일"),
    task_id: Optional[str] = Form(None, description="작업 ID (자동 생성)")
    ):
    """
    장난감 기부 가능 여부 확인을 위한 사진 전송
    
    - **image**: 이미지 파일 (JPG, PNG, JPEG)
    - **task_id**: 작업 ID (자동 생성)
    """
    try:
        # 이미지 파일 검증 (안전한 방식)
        if not image.content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미지 파일의 Content-Type을 확인할 수 없습니다."
            )
        
        # content_type이 string인지 확인하고 안전하게 검증
        content_type = str(image.content_type) if image.content_type else ""
        if not content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"이미지 파일만 업로드 가능합니다. 현재 타입: {content_type}"
            )
        
        # 이미지 데이터 읽기 (이미지 데이터를 bytes로 변환)
        image_data = await image.read()
        
        # 분석 작업 생성
        task_id = analyze_crud.create_analysis_task(image_data)
        
        # 백그라운드에서 분석 작업 시작
        analyze_crud.start_analysis_task(task_id)
        
        return analyze_schema.AnalyzeResponse(
            task_id=task_id,
            status="processing",
            message="분석이 시작되었습니다. 결과를 확인하려면 /result 엔드포인트를 사용하세요.",
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 작업 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/classify/result", response_model=analyze_schema.AnalyzeResultResponse)
async def get_classify_result(task_id: str = Query(..., description="분석 작업 ID")):
    """
    장난감 분류 결과 조회 Polling 요청
    
    - **task_id**: 분석 작업 ID
    """
    try:
        # 분석 작업 조회
        task = analyze_crud.get_analysis_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="분석 작업을 찾을 수 없습니다."
            )
        
        # 결과 구성
        result = None
        if task["status"] == "completed" and task["result"]:
            result = analyze_schema.ToyAnalysisResult(
                장난감_종류=task["result"].get("장난감 종류", ""),
                건전지_여부=task["result"].get("건전지 여부", ""),
                재료=task["result"].get("재료", ""),
                파손=task["result"].get("파손", ""),
                오염도=task["result"].get("오염도", ""),
                크기=task["result"].get("크기", ""),
                기부_가능_여부=task["result"].get("기부 가능 여부", ""),
                기부_불가_사유=task["result"].get("기부 불가 사유"),
                수리_분해=task["result"].get("수리/분해", ""),
                관찰사항=task["result"].get("관찰사항"),
                토큰_사용량=task["result"].get("토큰 사용량", {})
            )
        
        return analyze_schema.AnalyzeResultResponse(
            task_id=task["task_id"],
            status=task["status"],
            result=result,
            error=task["error"],
            created_at=task["created_at"],
            completed_at=task["completed_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"결과 조회 중 오류가 발생했습니다: {str(e)}"
        )
