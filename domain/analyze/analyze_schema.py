from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class AnalyzeResponse(BaseModel):
    """AI 분석 응답 스키마"""
    task_id: str
    status: str  # "processing", "completed", "failed"
    message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ToyAnalysisResult(BaseModel):
    """장난감 분석 결과 상세 스키마"""
    장난감_종류: str
    건전지_여부: str
    재료: str
    파손: str
    오염도: str
    크기: str
    기부_가능_여부: str
    기부_불가_사유: Optional[str] = None
    수리_분해: str
    관찰사항: Optional[str] = None
    토큰_사용량: Dict[str, int]

    class Config:
        from_attributes = True

class AnalyzeResultResponse(BaseModel):
    """분석 결과 응답 스키마"""
    task_id: str
    status: str
    result: Optional[ToyAnalysisResult] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True