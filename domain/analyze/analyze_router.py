from fastapi import APIRouter, UploadFile, File, HTTPException
from .ai_agent.supervisor_agent import SupervisorAgent
from .analyze_schema import AnalyzeResult

router = APIRouter()

def convert_result_keys(result):
    return {
        "toy_type": result.get("장난감 종류"),
        "battery": result.get("건전지 여부"),
        "material": result.get("재료"),
        "damage": result.get("파손"),
        "donate": result.get("기부 가능 여부") == "가능",
        "donate_reason": result.get("기부 불가 사유") or "",  # None이면 빈 문자열로
        "repair_or_disassemble": result.get("수리/분해"),
        "token_usage": result.get("토큰_사용량"),
    }


@router.post("/analyze", response_model=AnalyzeResult)
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()  # 이미지 바이트 읽기
    supervisor = SupervisorAgent()   # SupervisorAgent 인스턴스 생성
    result = supervisor.process(image_bytes)
    return convert_result_keys(result)