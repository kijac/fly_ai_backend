# 🎯 AI 에이전트 기부 가능 여부 평가 시스템

## 📁 폴더 구조
```
evaluation/
├── main_evaluation.py          # 메인 평가 스크립트
├── Data_to_evaluate.csv        # 평가할 데이터 (728개 이미지)
├── datasets/                   # 이미지 데이터셋
│   └── images/                # 실제 이미지 파일들
└── evaluation_results/         # 평가 결과 저장 폴더
```

## 🚀 평가 실행 방법

### 1. 기본 평가 실행
```bash
cd Ai_agent/ai_agent/evaluation
python main_evaluation.py
```

### 2. 평가 결과
- **JSON 결과**: `evaluation_results/improved_evaluation_Data_to_evaluate_[timestamp].json`
- **CSV 요약**: `evaluation_results/improved_evaluation_summary_Data_to_evaluate_[timestamp].csv`

## 📊 평가 지표

### 현재 성능 (50개 이미지 기준)
- **전체 정확도**: 54.00%
- **정확한 예측**: 27개
- **오류 예측**: 23개

### 주요 개선점
✅ **즉시 기부 불가 조건들 정확하게 판단**:
- 나무 소재
- 섬유/천 소재  
- 실리콘 소재
- 혼합 소재
- 부품 누락
- 심각한 파손

❌ **여전히 개선이 필요한 부분**:
- 혼합 소재 감지 정확도 (특히 플라스틱+천)
- 용도 불분명 장난감 식별
- 일부 부품 누락 감지

## 🔧 평가 설정 변경

### 테스트 이미지 수 조정
`main_evaluation.py`의 `max_test_count` 값을 변경:
```python
max_test_count = 50  # 원하는 이미지 수로 변경
```

### 평가 기준 변경
`supervisor_agent.py`의 `_judge_donation` 메서드에서 가중치 및 판단 로직 수정

## 📈 결과 분석

평가 결과는 다음 정보를 포함합니다:
- 각 이미지별 AI 예측 vs 실제 결과
- AI 판단 사유
- 토큰 사용량
- 정확도 통계
- 오류 케이스 상세 분석

## 🎯 다음 개선 목표

1. **혼합 소재 감지 정확도 향상**
2. **용도 불분명 장난감 식별 강화**
3. **부품 누락 감지 정확도 개선**
4. **전체 정확도 70% 이상 달성**
