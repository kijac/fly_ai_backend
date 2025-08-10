import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from ai_agent.supervisor_agent import SupervisorAgent

# 분석 작업 저장소 (실제로는 데이터베이스에 저장해야 함)
analysis_tasks: Dict[str, Dict[str, Any]] = {}

def create_analysis_task(image_data: bytes) -> str:
    """AI 분석 작업 생성"""
    task_id = str(uuid.uuid4())
    
    # 작업 정보 저장
    analysis_tasks[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "image_data": image_data,
        "result": None,
        "error": None,
        "created_at": datetime.now(),
        "completed_at": None
    }
    
    return task_id

def get_analysis_task(task_id: str) -> Optional[Dict[str, Any]]:
    """분석 작업 조회"""
    return analysis_tasks.get(task_id)

def update_analysis_result(task_id: str, result: Dict[str, Any], error: Optional[str] = None):
    """분석 결과 업데이트"""
    if task_id in analysis_tasks:
        analysis_tasks[task_id]["status"] = "completed" if error is None else "failed"
        analysis_tasks[task_id]["result"] = result
        analysis_tasks[task_id]["error"] = error
        analysis_tasks[task_id]["completed_at"] = datetime.now()

def process_analysis_task(task_id: str):
    """AI 분석 작업 처리"""
    try:
        task = analysis_tasks.get(task_id)
        if not task:
            return
        
        # 이미지 데이터는 이미 bytes 타입
        image_bytes = task["image_data"]
        
        # AI 에이전트로 분석
        supervisor = SupervisorAgent()
        result = supervisor.process(image_bytes)
        
        # 결과 업데이트
        update_analysis_result(task_id, result)
        
    except Exception as e:
        # 에러 발생 시
        print(f"분석 작업 오류 (task_id: {task_id}): {e}")
        update_analysis_result(task_id, None, str(e))

def start_analysis_task(task_id: str):
    """분석 작업 시작 (백그라운드에서 실행)"""
    # 별도 스레드에서 분석 실행
    thread = threading.Thread(target=process_analysis_task, args=(task_id,))
    thread.daemon = True  # 메인 스레드 종료 시 함께 종료
    thread.start()
