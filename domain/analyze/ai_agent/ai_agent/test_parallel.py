#!/usr/bin/env python3
"""
병렬 처리 vs 순차 처리 성능 테스트 스크립트
"""

import time
import os
from supervisor_agent import SupervisorAgent

def create_dummy_image():
    """테스트용 더미 이미지 생성 (실제로는 파일에서 읽어야 함)"""
    # 실제 테스트에서는 이미지 파일을 읽어야 함
    return b"dummy_image_data_for_testing"

def test_processing_modes():
    """병렬 처리와 순차 처리의 성능 비교"""
    print("🧸 장난감 기부 판별 AI - 병렬 처리 성능 테스트")
    print("=" * 60)
    
    # 더미 이미지 생성 (실제 테스트에서는 실제 이미지 파일 사용)
    dummy_image = create_dummy_image()
    
    supervisor = SupervisorAgent()
    
    # 순차 처리 테스트
    print("\n📊 순차 처리 테스트 시작...")
    start_time = time.time()
    try:
        sequential_result = supervisor.process_sequential(dummy_image, dummy_image)
        sequential_time = time.time() - start_time
        print(f"✅ 순차 처리 완료: {sequential_time:.2f}초")
        print(f"   결과: {sequential_result}")
    except Exception as e:
        print(f"❌ 순차 처리 실패: {e}")
        sequential_time = float('inf')
    
    # 병렬 처리 테스트
    print("\n🚀 병렬 처리 테스트 시작...")
    start_time = time.time()
    try:
        parallel_result = supervisor.process(dummy_image, dummy_image)
        parallel_time = time.time() - start_time
        print(f"✅ 병렬 처리 완료: {parallel_time:.2f}초")
        print(f"   결과: {parallel_result}")
    except Exception as e:
        print(f"❌ 병렬 처리 실패: {e}")
        parallel_time = float('inf')
    
    # 성능 비교
    print("\n📈 성능 비교 결과")
    print("-" * 40)
    
    if sequential_time != float('inf') and parallel_time != float('inf'):
        if parallel_time < sequential_time:
            speedup = sequential_time / parallel_time
            print(f"🚀 병렬 처리가 {speedup:.2f}배 빠릅니다!")
            print(f"   순차 처리: {sequential_time:.2f}초")
            print(f"   병렬 처리: {parallel_time:.2f}초")
            print(f"   시간 절약: {sequential_time - parallel_time:.2f}초")
        else:
            print(f"⚠️  병렬 처리가 순차 처리보다 {parallel_time - sequential_time:.2f}초 느립니다")
            print(f"   순차 처리: {sequential_time:.2f}초")
            print(f"   병렬 처리: {parallel_time:.2f}초")
    else:
        print("❌ 테스트 중 오류가 발생했습니다")
    
    print("\n💡 참고사항:")
    print("- 실제 이미지로 테스트하면 더 정확한 성능 측정이 가능합니다")
    print("- 네트워크 지연이나 API 응답 시간에 따라 결과가 달라질 수 있습니다")
    print("- 병렬 처리는 4개의 AI 에이전트를 동시에 실행합니다")

if __name__ == "__main__":
    # 환경변수 확인
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   실제 API 호출 테스트를 위해서는 API 키가 필요합니다.")
        print("   .env 파일에 GEMINI_API_KEY=your_key_here 를 추가하세요.")
    
    test_processing_modes()
