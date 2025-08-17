# 🎯 100개 이미지 평가 시스템 사용법

이 가이드는 `Img_data` 폴더에 있는 100개의 `Img_*_front.png` 이미지를 사용하여 AI 에이전트의 정확도를 평가하는 방법을 설명합니다.

## 📁 **파일 구조**

```
Ai_agent/ai_agent/evaluation/
├── 📁 datasets/
│   ├── 📁 images/                    # 여기에 100개 front 이미지들 넣기
│   │   ├── Img_0000_front.png
│   │   ├── Img_0001_front.png
│   │   ├── Img_0002_front.png
│   │   └── ... (총 100개)
│   │
│   ├── 100_images_dataset.json       # 자동 생성된 데이터셋
│   ├── ground_truth_template.json    # Ground Truth 작성 템플릿
│   └── toy_evaluation_dataset.json   # 기존 샘플 데이터셋
│
├── 📁 results/                       # 평가 결과 저장
├── generate_100_dataset.py           # 100개 데이터셋 자동 생성
├── evaluation_system.py              # 평가 시스템
├── run_evaluation.py                 # 평가 실행
└── 100_IMAGES_README.md              # 이 파일
```

## 🚀 **사용 단계**

### **1단계: 100개 데이터셋 자동 생성**
```bash
cd Ai_agent/ai_agent/evaluation
python generate_100_dataset.py
```

이 스크립트는 다음을 생성합니다:
- `100_images_dataset.json`: 100개 이미지용 기본 데이터셋
- `ground_truth_template.json`: Ground Truth 작성 가이드

### **2단계: 이미지 파일 복사**
```bash
# Img_data 폴더에서 front 이미지들만 복사
cp /path/to/Img_data/Img_*_front.png datasets/images/
```

### **3단계: Ground Truth 데이터 작성**
`ground_truth_template.json`을 참고하여 실제 정답 데이터를 작성합니다.

**Ground Truth 항목들:**
- `toy_type`: 장난감 종류 (피규어, 자동차 장난감, 변신 로봇, 블록, 공, 인형 등)
- `battery`: 건전지 여부 (건전지, 비건전지, 불명)
- `size`: 크기 (작음, 중간, 큼, 불명)
- `material`: 재료 (플라스틱, 금속, 나무, 천, 섬유, 고무, 종이, 기타)
- `damage`: 파손 상태 (없음, 경미한 파손, 미세한 파손, 심각한 파손, 불명)
- `soil`: 오염도 (깨끗, 약간 더러움, 더러움, 불명)
- `donation_possible`: 기부 가능 여부 (true/false)
- `donation_reason`: 기부 가능/불가 사유

### **4단계: 평가 실행**
```bash
# 100개 이미지로 평가 실행
python run_evaluation.py --dataset datasets/100_images_dataset.json --visualize
```

## 📊 **예상 결과**

100개 이미지 평가 시:
- **처리 시간**: 약 10-15분 (이미지당 6-9초)
- **토큰 사용량**: 약 15,000-20,000 토큰
- **정확도**: 각 에이전트별 상세 성능 지표
- **시각화**: 막대 그래프로 성능 비교

## 💡 **팁과 주의사항**

### **1. 데이터 품질**
- **이미지 품질**: 명확하고 일관된 이미지 사용
- **Ground Truth**: 전문가가 검증한 정확한 라벨
- **다양성**: 기부 가능/불가능, 다양한 장난감 종류 포함

### **2. 비용 관리**
- **토큰 사용량**: 100개 이미지당 약 $0.15-0.20 예상
- **배치 처리**: 필요시 작은 배치로 나누어 평가

### **3. 성능 최적화**
- **병렬 처리**: 4개 에이전트가 동시에 실행
- **타임아웃**: 30초 내 응답 없으면 순차 처리로 전환

## 🔧 **문제 해결**

### **일반적인 오류**
1. **이미지 파일을 찾을 수 없음**
   - 이미지 경로 확인
   - 파일명 대소문자 확인 (Img_ vs img_)

2. **JSON 파싱 오류**
   - Ground Truth 데이터 형식 검증
   - 특수문자 이스케이프 처리

3. **API 키 오류**
   - `.env` 파일에 OPENAI_API_KEY 설정 확인

### **디버깅**
```bash
# 상세한 오류 정보 출력
python run_evaluation.py --dataset datasets/100_images_dataset.json 2>&1 | tee evaluation_100.log
```

## 📈 **결과 분석**

### **개별 에이전트 성능**
- **TypeAgent**: 장난감 종류, 건전지 여부, 크기 정확도
- **MaterialAgent**: 재료 분석 정확도
- **DamageAgent**: 파손 상태 분석 정확도
- **SoilAgent**: 오염도 분석 정확도

### **통합 성능**
- **SupervisorAgent**: 전체 기부 판단 정확도
- **평균 정확도**: 모든 에이전트의 평균 성능

## 🎯 **성능 개선**

### **정확도가 낮은 경우**
1. **프롬프트 최적화**: system prompt 개선
2. **데이터 품질**: Ground Truth 정확성 향상
3. **이미지 품질**: 더 명확한 이미지 사용

### **비용 최적화**
1. **샘플링**: 대표적인 이미지만 선별
2. **배치 크기**: 적절한 배치 크기로 분할
3. **모델 선택**: 필요시 더 저렴한 모델 사용

## 📞 **지원**

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해 주세요.
