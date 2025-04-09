from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import tryon, mediapipe, segmentation, diffusion

app = FastAPI(title="TryOn API", description="Virtual Try-On API using AI models")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(tryon.router, prefix="/api/tryon", tags=["tryon"])
app.include_router(mediapipe.router, prefix="/api/mediapipe", tags=["mediapipe"])
app.include_router(segmentation.router, prefix="/api/segmentation", tags=["segmentation"])
app.include_router(diffusion.router, prefix="/api/diffusion", tags=["diffusion"])

@app.get("/")
async def root():
    return {"message": "Welcome to TryOn API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)