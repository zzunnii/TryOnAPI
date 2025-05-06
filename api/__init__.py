"""
TryOnAPI 패키지 초기화
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api.core.config import get_settings
from api.routers import preprocessing, tryon, clothing, user

settings = get_settings()

def create_app():
    """FastAPI 애플리케이션 생성"""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="가상 피팅 API (DensePose 외부 API 연동 버전)",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(preprocessing.router, prefix="/api/v1/preprocessing", tags=["전처리"])
    app.include_router(tryon.router, prefix="/api/v1/tryon", tags=["가상 피팅"])
    app.include_router(clothing.router, prefix="/api/v1/clothing", tags=["의류"])
    app.include_router(user.router, prefix="/api/v1/user", tags=["사용자"])

    # 정적 파일 디렉토리 설정
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")

    # 앱 시작 이벤트
    @app.on_event("startup")
    async def startup_event():
        """애플리케이션 시작 시 실행"""
        print(f"{settings.APP_NAME} v{settings.APP_VERSION} 시작")
        print(f"모드: {'디버그' if settings.DEBUG else '운영'}")
        print(f"장치: {settings.DEVICE}")
        print(f"DensePose API URL: {settings.DENSEPOSE_API_URL}")

    # 앱 종료 이벤트
    @app.on_event("shutdown")
    async def shutdown_event():
        """애플리케이션 종료 시 실행"""
        print(f"{settings.APP_NAME} 종료")

    return app