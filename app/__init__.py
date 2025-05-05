"""
TryOnAPI FastAPI 애플리케이션
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings

# 설정 가져오기
settings = get_settings()

def create_app():
    """애플리케이션 생성 함수"""
    # FastAPI 앱 생성
    app = FastAPI(
        title="TryOnAPI",
        description="가상 의류 피팅을 위한 API 서버",
        version="1.0.0",
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 모든 오리진 허용 (실제 운영에서는 제한하는 것이 좋음)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 정적 파일 서빙을 위한 디렉토리 생성
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    
    # 정적 파일 마운트
    app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")
    
    # 라우터 등록
    from app.routers import human_parsing, segmentation, tryon, mediapipe
    
    # 라우터 포함
    app.include_router(segmentation.router, prefix=settings.API_V1_STR, tags=["Segmentation"])
    app.include_router(human_parsing.router, prefix=settings.API_V1_STR, tags=["Human Parsing"])
    app.include_router(tryon.router, prefix=settings.API_V1_STR + "/tryon", tags=["Virtual Try-On"])
    app.include_router(mediapipe.router, prefix=settings.API_V1_STR + "/mediapipe", tags=["MediaPipe"])
    
    # 루트 엔드포인트
    @app.get("/")
    async def root():
        return {"message": "TryOnAPI 서버가 실행 중입니다."}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "tryon-api"}
    
    return app
