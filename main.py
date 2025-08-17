from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from domain.user import user_router
from domain.toy_stock import toy_stock_router
from domain.analyze import analyze_router
from domain.donor_requests import donor_requests_router


app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "*",  # 모든 출처 허용 (Flutter 앱 연결용)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from domain.analyze.analyze_router import router as analyze_router

# 테스트용 이미지 파일 서빙 설정
app.mount("/api/images", StaticFiles(directory="toypics"), name="images")

#라우터 탑재 템플릿
app.include_router(user_router.router)
app.include_router(toy_stock_router.router)
app.include_router(analyze_router, prefix="/api")
app.include_router(donor_requests_router.router)