#!/usr/bin/env python3
"""
새로운 AI 모델 테스트 스크립트
"""

import os
import sys
import tempfile
from pathlib import Path

# AI 모델 경로 추가
ai_agent_path = Path(__file__).parent / "domain" / "analyze" / "ai_agent"
sys.path.append(str(ai_agent_path))

def test_ai_model():
    """AI 모델이 정상적으로 로드되는지 테스트"""
    try:
        print("🔍 AI 모델 테스트 시작...")
        
        # 작업 디렉토리 변경
        original_cwd = os.getcwd()
        os.chdir(ai_agent_path)
        
        try:
            # 모듈 import 테스트
            print("📦 모듈 import 테스트...")
            from ai_agent_output import run_full_pipeline
            from ai_agent.supervisor_agent import SupervisorAgent
            from predict import run_sameitem_price
            
            print("✅ 모든 모듈이 정상적으로 import되었습니다!")
            
            # 간단한 테스트 이미지 생성 (1x1 픽셀)
            print("🖼️ 테스트 이미지 생성...")
            from PIL import Image
            import numpy as np
            
            # 테스트용 이미지 생성
            test_image = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
            test_path = tempfile.mktemp(suffix=".jpg")
            test_image.save(test_path)
            
            print(f"📁 테스트 이미지 생성: {test_path}")
            
            # AI 모델 실행 테스트 (기본 기능만)
            print("🤖 AI 모델 실행 테스트...")
            
            # SupervisorAgent 테스트
            supervisor = SupervisorAgent()
            print("✅ SupervisorAgent 초기화 성공!")
            
            # 임시 파일 정리
            os.unlink(test_path)
            
            print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            return True
            
        finally:
            # 작업 디렉토리 복원
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ai_model()
    sys.exit(0 if success else 1)
