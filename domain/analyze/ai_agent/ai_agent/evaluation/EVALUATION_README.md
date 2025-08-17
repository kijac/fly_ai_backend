# AI 에이전트 정확도 평가 시스템

이 시스템은 AI 에이전트들이 장난감 이미지를 분석할 때의 정확도를 측정하고 평가하는 도구입니다.

## 🚀 주요 기능

- **개별 에이전트 평가**: TypeAgent, MaterialAgent, DamageAgent, SoilAgent 각각의 성능 측정
- **통합 평가**: SupervisorAgent의 전체 성능 측정
- **정확도 계산**: 각 항목별 정확도 및 전체 정확도 계산
- **토큰 사용량 추적**: OpenAI API 토큰 사용량 모니터링
- **결과 시각화**: 막대 그래프를 통한 성능 비교
- **결과 저장**: JSON, CSV 형태로 상세 결과 저장

## 📁 파일 구조

```
ai_agent/
├── evaluation_system.py      # 핵심 평가 시스템
├── run_evaluation.py         # 평가 실행 스크립트
├── sample_dataset.json       # 샘플 평가 데이터셋
├── EVALUATION_README.md      # 이 파일
└── node_agents/             # 개별 AI 에이전트들
    ├── type_agent.py
    ├── material_agent.py
    ├── damage_agent.py
    └── soil_agent.py
```

## 🛠️ 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install pandas numpy matplotlib seaborn openai python-dotenv
```

### 2. 환경 변수 설정

`.env` 파일에 OpenAI API 키를 설정하세요:

```bash
OPENAI_API_KEY=your_api_key_here
```

## 📊 평가 데이터셋 형식

평가용 데이터셋은 JSON 또는 CSV 형식으로 준비해야 합니다.

### JSON 형식 예시:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "ground_truth": {
      "toy_type": "자동차 장난감",
      "battery": "건전지",
      "size": "중간",
      "material": "플라스틱",
      "damage": "없음",
      "soil": "깨끗",
      "donation_possible": true,
      "donation_reason": "플라스틱 건전지 장난감, 상태 양호"
    }
  }
]
```

### CSV 형식 예시:

```csv
image_path,toy_type,battery,size,material,damage,soil,donation_possible,donation_reason
path/to/image.jpg,자동차 장난감,건전지,중간,플라스틱,없음,깨끗,true,플라스틱 건전지 장난감 상태 양호
```

## 🎯 평가 실행 방법

### 기본 실행

```bash
python run_evaluation.py --dataset sample_dataset.json
```

### 옵션과 함께 실행

```bash
# 시각화 포함하여 실행
python run_evaluation.py --dataset sample_dataset.json --visualize

# 결과 저장 디렉토리 지정
python run_evaluation.py --dataset sample_dataset.json --output-dir my_results

# 모든 옵션 사용
python run_evaluation.py --dataset sample_dataset.json --visualize --output-dir my_results
```

## 📈 평가 결과

### 1. 콘솔 출력

평가가 완료되면 다음과 같은 요약 정보가 출력됩니다:

```
🎯 AI 에이전트 정확도 평가 결과
============================================================
📊 평가 일시: 2024-01-15T10:30:00
📁 총 샘플 수: 10
🔍 평균 정확도: 0.850
💾 총 토큰 사용량: 15,420

📈 개별 에이전트 성능:
----------------------------------------
✅ type_agent      | 정확도: 0.900 | 처리: 10/10
✅ material_agent  | 정확도: 0.800 | 처리: 10/10
⚠️ damage_agent    | 정확도: 0.700 | 처리: 10/10
✅ soil_agent      | 정확도: 0.900 | 처리: 10/10
✅ supervisor      | 정확도: 0.850 | 처리: 10/10
============================================================
```

### 2. 파일 저장

평가 결과는 다음 파일들로 저장됩니다:

- **JSON 파일**: `evaluation_results_YYYYMMDD_HHMMSS.json` - 상세한 평가 결과
- **CSV 파일**: `evaluation_summary_YYYYMMDD_HHMMSS.csv` - 요약된 성능 지표
- **그래프**: `accuracy_comparison_YYYYMMDD_HHMMSS.png` - 시각화 결과 (--visualize 옵션 사용 시)

## 🔍 정확도 계산 방법

### 개별 에이전트 정확도

- **TypeAgent**: toy_type, battery, size 3개 항목의 정확도 평균
- **MaterialAgent**: material 항목 정확도
- **DamageAgent**: damage 항목 정확도  
- **SoilAgent**: soil 항목 정확도

### 슈퍼바이저 정확도

6개 주요 항목(장난감 종류, 건전지 여부, 재료, 파손, 오염도, 기부 가능 여부)의 정확도 평균

## 💡 사용 팁

### 1. 데이터셋 준비

- **다양한 케이스 포함**: 기부 가능/불가능, 다양한 장난감 종류, 파손 상태 등
- **이미지 품질**: 명확하고 일관된 이미지 사용
- **정답 라벨**: 정확하고 일관된 ground truth 데이터 준비

### 2. 평가 실행

- **작은 데이터셋으로 테스트**: 먼저 몇 개 샘플로 시스템 테스트
- **토큰 사용량 모니터링**: 대용량 데이터셋 평가 시 비용 고려
- **결과 분석**: 개별 에이전트별 성능 차이 분석

### 3. 성능 개선

- **약점 파악**: 정확도가 낮은 에이전트 식별
- **프롬프트 최적화**: system prompt 개선
- **데이터 품질**: ground truth 데이터 정확성 향상

## 🚨 주의사항

1. **API 비용**: OpenAI API 사용량에 따른 비용 발생
2. **이미지 경로**: 데이터셋의 이미지 경로가 올바른지 확인
3. **메모리 사용**: 대용량 데이터셋 처리 시 메모리 사용량 고려
4. **네트워크**: 안정적인 인터넷 연결 필요

## 🔧 문제 해결

### 일반적인 오류

1. **이미지 파일을 찾을 수 없음**: 이미지 경로 확인
2. **JSON 파싱 오류**: AI 응답 형식 검증
3. **API 키 오류**: 환경 변수 설정 확인
4. **메모리 부족**: 데이터셋 크기 조정

### 디버깅

```bash
# 상세한 오류 정보 출력
python run_evaluation.py --dataset sample_dataset.json 2>&1 | tee evaluation.log
```

## 📞 지원

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해 주세요.
