"""
사용자 이미지 관련 라우터
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Optional

from api.core.config import get_settings
from api.utils.image_utils import save_image

router = APIRouter()
settings = get_settings()

@router.post("/upload-image")
async def upload_user_image(
    file: UploadFile = File(...),
):
    """
    사용자 이미지 업로드
    
    Parameters:
    - file: 업로드할 이미지 파일
    
    Returns:
    - file_id: 이미지 ID
    - filename: 원본 파일명
    - file_path: 저장된 파일 경로
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다. JPG 또는 PNG만 지원합니다.")
        
        # 저장 디렉토리 확인 및 생성
        file_id = str(uuid.uuid4())
        save_dir = settings.USER_IMAGES_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일 저장 경로
        file_path = os.path.join(save_dir, f"{file_id}{file_ext}")
        
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 업로드 중 오류: {str(e)}")

@router.get("/images/{image_id}")
async def get_user_image(image_id: str):
    """
    사용자 이미지 조회
    
    Parameters:
    - image_id: 이미지 ID
    
    Returns:
    - 이미지 파일
    """
    try:
        # 이미지 파일 경로 탐색
        file_dir = settings.USER_IMAGES_DIR
        matching_files = [f for f in os.listdir(file_dir) if f.startswith(image_id)]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail=f"이미지 ID를 찾을 수 없습니다: {image_id}")
        
        file_path = os.path.join(file_dir, matching_files[0])
        
        return FileResponse(file_path, media_type="image/jpeg")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 조회 중 오류: {str(e)}")

@router.delete("/images/{image_id}")
async def delete_user_image(image_id: str):
    """
    사용자 이미지 삭제
    
    Parameters:
    - image_id: 이미지 ID
    
    Returns:
    - 삭제 상태
    """
    try:
        # 이미지 파일 경로 탐색
        file_dir = settings.USER_IMAGES_DIR
        matching_files = [f for f in os.listdir(file_dir) if f.startswith(image_id)]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail=f"이미지 ID를 찾을 수 없습니다: {image_id}")
        
        # 파일 삭제
        for file_name in matching_files:
            file_path = os.path.join(file_dir, file_name)
            os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"이미지 ID {image_id}가 삭제되었습니다.",
            "deleted_files": matching_files
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 삭제 중 오류: {str(e)}")