# Toy Donation AI Backend

장난감 기부 판정을 위한 AI 기반 백엔드 서버입니다.

## 🚀 주요 기능

- **AI 기반 장난감 분석**: CLIP 모델과 Gemini AI를 활용한 장난감 타입, 재질, 손상도, 오염도 분석
- **병렬 처리**: 4개 AI 에이전트의 동시 실행으로 빠른 분석 속도
- **RESTful API**: FastAPI 기반의 REST API 제공
- **데이터베이스 연동**: MySQL 데이터베이스와 연동

## 📋 기술 스택

- **Backend**: FastAPI, Python 3.8+
- **Database**: MySQL
- **AI Models**: 
  - CLIP (OpenAI) - 이미지 유사도 검색
  - Gemini 2.0 Flash (Google) - 장난감 분석
- **ORM**: SQLAlchemy
- **Image Processing**: PIL, OpenCV

## 🛠️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone [repository-url]
cd myapi
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=myapi_db
DB_USER=root
DB_PASSWORD=YOUR_PASSWORD

# auth
SECRET_KEY=900599dd55a7a691e0e116e26233bc31c15d3ee98affa39601ba8760cb1319e1
ACCESS_TOKEN_EXPIRE_MINUTES=1440

ACCESS_TOKEN_EXPIRE_MINUTES = 60*24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl = "/api/user/login")


OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here



# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 5. 데이터베이스 설정
```bash
# 데이터베이스 마이그레이션
alembic upgrade head
```

### 6. 서버 실행
```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 🔧 환경변수 상세 설명

### 필수 환경변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `DATABASE_URL` | MySQL 데이터베이스 연결 URL | `mysql+pymysql://user:pass@localhost:3306/toydb` |
| `GOOGLE_API_KEY` | Google AI API 키 (Gemini 사용) | `AIzaSyC...` |

### 선택적 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `HOST` | 서버 호스트 | `0.0.0.0` |
| `PORT` | 서버 포트 | `8000` |
| `DEBUG` | 디버그 모드 | `True` |

## 📡 API 엔드포인트

### 분석 API
- `POST /api/analyze` - 장난감 이미지 분석
  - **Request**: 4개 이미지 (front, left, rear, right)
  - **Response**: 분석 결과 (타입, 재질, 손상도, 오염도)

### 관리자 API
- `POST /api/admin/login` - 관리자 로그인
- `GET /api/admin/users` - 사용자 목록 조회
- `POST /api/admin/users` - 사용자 생성

### 기부 요청 API
- `GET /api/donor-requests` - 기부 요청 목록
- `POST /api/donor-requests` - 기부 요청 생성
- `PUT /api/donor-requests/{id}` - 기부 요청 수정

## 🤖 AI 모델 구성

### 에이전트 구조
- **TypeAgent**: 장난감 타입, 배터리, 크기 분류
- **MaterialAgent**: 재질 분석 (플라스틱, 금속, 목재 등)
- **DamageAgent**: 손상도 및 부품 누락 분석
- **SoilAgent**: 오염도 분석

### 성능 최적화
- **CLIP 모델**: 싱글톤 패턴으로 한 번만 로딩
- **병렬 처리**: 4개 에이전트 동시 실행
- **GPU 가속**: CUDA 지원 (가능한 경우)

## 📁 프로젝트 구조

```
myapi/
├── main.py                 # 메인 서버 파일
├── database.py             # 데이터베이스 설정
├── model.py                # 데이터베이스 모델
├── requirements.txt        # Python 의존성
├── .env                    # 환경변수 (Git에서 제외)
├── domain/                 # 비즈니스 로직
│   ├── analyze/           # AI 분석 모듈
│   ├── user/              # 사용자 관리
│   ├── admin/             # 관리자 기능
│   └── donor_requests/    # 기부 요청 관리
└── toypics/               # 테스트 이미지 (Git에서 제외)
```

## 🔍 디버깅

### 로그 확인
서버 실행 시 다음과 같은 로그를 확인할 수 있습니다:
```
🚀 AI 분석 시작: [이미지 경로]
🔍 CLIP 유사도 검색 완료: X.XX초
🚀 모든 에이전트 병렬 실행 시작...
✅ TypeAgent 완료 (병렬)
✅ MaterialAgent 완료 (병렬)
✅ DamageAgent 완료 (병렬)
✅ SoilAgent 완료 (병렬)
🤖 에이전트 API 호출 완료: X.XX초
✅ AI 분석 완료: X.XX초
```

## 🚨 주의사항

1. **API 키 보안**: `.env` 파일은 절대 Git에 커밋하지 마세요
2. **GPU 메모리**: CLIP 모델은 GPU 메모리를 사용합니다
3. **이미지 크기**: 업로드 이미지는 자동으로 최적화됩니다
4. **타임아웃**: AI 분석은 최대 30초까지 걸릴 수 있습니다

## 📞 지원

문제가 발생하면 이슈를 등록해주세요.
