@router.get("/result/{result_id}")
async def get_segmentation_result(result_id: str):
    """
    세그멘테이션 결과 이미지 반환
    """
    try:
        # 결과 이미지 경로 찾기
        result_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{result_id}_segmentation.jpg")
        
        # 파일이 존재하는지 확인
        if not os.path.exists(result_path):
            # 다른 폴더도 확인 (agnostic, no_background)
            alternative_paths = [
                os.path.join(settings.OUTPUT_DIR, "agnostic", f"{result_id}_agnostic.jpg"),
                os.path.join(settings.OUTPUT_DIR, "no_background", f"{result_id}_nobg.jpg")
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    result_path = path
                    break
            else:
                raise HTTPException(status_code=404, detail=f"Result with ID {result_id} not found")
        
        # 파일 반환
        return FileResponse(result_path)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any
import os
import uuid

from app.core.config import get_settings
from app.services.segmentation_service import segment_image, remove_background, create_agnostic_image
from app.models.image import ImageResponse

router = APIRouter()
settings = get_settings()

@router.post("/process", response_model=ImageResponse)
async def segment_image_endpoint(
    image_id: str,
):
    """
    이미지 세그멘테이션 수행 (Class 20)
    """
    try:
        # 이미지 ID로 저장된 이미지 경로 찾기
        image_path = os.path.join(settings.OUTPUT_DIR, "uploads", f"{image_id}.jpg")
        
        # 이미지가 존재하는지 확인
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
        
        # 세그멘테이션 수행
        result = segment_image(image_path)
        
        # 세그멘테이션 결과에서 파일 이름만 추출
        segmentation_filename = os.path.basename(result['segmentation_path'])
        mask_filename = os.path.basename(result['mask_path'])
        
        # 결과 반환
        return {
            "id": str(uuid.uuid4()),
            "original_id": image_id,
            "segmentation_path": f"/output/segmentation/{segmentation_filename}",
            "mask_path": f"/output/segmentation/{mask_filename}",
            "message": "Segmentation completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove-background", response_model=ImageResponse)
async def remove_background_endpoint(
    segmentation_id: str,
):
    """
    세그멘테이션 결과를 사용하여 배경 제거
    """
    try:
        # 세그멘테이션 결과 경로 찾기
        segmentation_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{segmentation_id}_segmentation.jpg")
        mask_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{segmentation_id}_mask.png")
        
        # 파일이 존재하는지 확인
        if not os.path.exists(segmentation_path) or not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail=f"Segmentation result with ID {segmentation_id} not found")
        
        # 배경 제거 수행
        result_path = remove_background({
            'segmentation_path': segmentation_path,
            'mask_path': mask_path
        })
        
        # 결과 파일 이름 추출
        result_filename = os.path.basename(result_path)
        
        # 결과 반환
        return {
            "id": str(uuid.uuid4()),
            "original_id": segmentation_id,
            "image_path": f"/output/no_background/{result_filename}",
            "message": "Background removal completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-agnostic", response_model=ImageResponse)
async def create_agnostic_image_endpoint(
    segmentation_id: str,
    keypoints_id: str,
):
    """
    세그멘테이션 결과와 키포인트를 사용하여 Agnostic 이미지 생성
    """
    try:
        # 세그멘테이션 결과 경로 찾기
        segmentation_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{segmentation_id}_segmentation.jpg")
        
        # 키포인트 결과 경로 찾기
        keypoints_path = os.path.join(settings.OUTPUT_DIR, "keypoints", f"{keypoints_id}.json")
        
        # 파일이 존재하는지 확인
        if not os.path.exists(segmentation_path):
            raise HTTPException(status_code=404, detail=f"Segmentation result with ID {segmentation_id} not found")
        
        if not os.path.exists(keypoints_path):
            raise HTTPException(status_code=404, detail=f"Keypoints result with ID {keypoints_id} not found")
        
        # Agnostic 이미지 생성
        result_path = create_agnostic_image(segmentation_path, keypoints_path)
        
        # 결과 파일 이름 추출
        result_filename = os.path.basename(result_path)
        
        # 결과 반환
        return {
            "id": str(uuid.uuid4()),
            "original_id": segmentation_id,
            "image_path": f"/output/agnostic/{result_filename}",
            "message": "Agnostic image created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))