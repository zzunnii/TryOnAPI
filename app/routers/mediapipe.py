from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os

from app.core.config import get_settings
from app.services.mediapipe_service import extract_keypoints

router = APIRouter()
settings = get_settings()

@router.post("/keypoints")
async def get_keypoints(
    image_id: str,
):
    """
    MediaPipe를 사용하여 이미지에서 키포인트 추출
    Class 17 키포인트 추출
    """
    try:
        # 구현 예정: image_id로 저장된 이미지 찾아서 MediaPipe로 키포인트 추출
        # keypoints = extract_keypoints(image_path)
        
        return {"message": "MediaPipe keypoints extraction will be implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))