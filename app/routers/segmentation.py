from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Optional
import os

from app.core.config import get_settings
from app.services.segmentation_service import segment_image, remove_background

router = APIRouter()
settings = get_settings()

@router.post("/process")
async def segment_image_endpoint(
    image_id: str,
):
    """
    이미지 세그멘테이션 수행 (Class 20)
    """
    try:
        # 구현 예정: image_id로 저장된 이미지 찾아서 세그멘테이션 수행
        # result_path = segment_image(image_path)
        
        return {"message": "Image segmentation will be implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove-background")
async def remove_background_endpoint(
    segmentation_id: str,
):
    """
    세그멘테이션 결과를 사용하여 배경 제거
    """
    try:
        # 구현 예정: segmentation_id로 저장된 세그멘테이션 결과 찾아서 배경 제거
        # result_path = remove_background(segmentation_path)
        
        return {"message": "Background removal will be implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-agnostic")
async def create_agnostic_image(
    segmentation_id: str,
    keypoints_id: str,
):
    """
    세그멘테이션 결과와 키포인트를 사용하여 Agnostic 이미지 생성
    """
    try:
        # 구현 예정: segmentation_id와 keypoints_id를 사용하여 Agnostic 이미지 생성
        
        return {"message": "Agnostic image creation will be implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))