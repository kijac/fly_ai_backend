from fastapi import APIRouter, HTTPException, Query, File, UploadFile, Form
from starlette import status
from datetime import datetime
from typing import Optional, Dict

from domain.analyze import analyze_schema, analyze_crud

router = APIRouter(
    prefix="/api/analyze",
)

@router.post("/classify", response_model=analyze_schema.AnalyzeResponse)
async def classify_toy(
    front: UploadFile = File(..., description="정면"),
    rear: UploadFile = File(..., description="후면"),
    left: UploadFile = File(..., description="좌측"),
    right: UploadFile = File(..., description="우측"),
    task_id: Optional[str] = Form(None, description="작업 ID (자동 생성)")
    ):
    """
    장난감 기부 가능 여부 확인을 위한 4면(전/후/좌/우) 이미지 전송
    - **task_id**: 작업 ID (자동 생성)
    """
    try:
        # 파일 검증
        analyze_crud._validate_image(front, "front")
        analyze_crud._validate_image(rear, "rear")
        analyze_crud._validate_image(left, "left")
        analyze_crud._validate_image(right, "right")
        
        # 이미지 데이터 읽기 (이미지 데이터를 bytes로 변환)
        images: Dict[str, bytes] = {
            "front": await front.read(),
            "rear": await rear.read(),
            "left": await left.read(),
            "right": await right.read(),
        }
        
        # 분석 작업 생성
        task_id = analyze_crud.create_analysis_task(images)
        
        # 백그라운드에서 분석 작업 시작
        analyze_crud.start_analysis_task(task_id)
        
        return analyze_schema.AnalyzeResponse(
            task_id=task_id,
            status="processing",
            message="분석이 시작되었습니다. 결과를 확인하려면 classify/result 엔드포인트를 사용하세요.",
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
