from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Optional
import uuid
import os

from app.services.image_service import save_upload_file
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

@router.post("/complete")
async def try_on_complete_process(
    user_image: UploadFile = File(...),
    target_clothing: UploadFile = File(...),
    output_size: Optional[str] = Form("1024")
):
    """
    전체 가상 착용 과정을 한번에 처리하는 엔드포인트
    1. 사용자 이미지 받기
    2. MediaPipe 키포인트 추출
    3. 이미지 세그멘테이션
    4. 배경 제거
    5. Agnostic 이미지 생성
    6. Diffusion 모델 처리 (선택한 크기)
    7. 결과 반환
    """
    
    try:
        # 구현 예정
        return {"message": "Complete try-on process will be implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/user-image")
async def upload_user_image(
    file: UploadFile = File(...)
):
    """
    사용자 이미지 업로드
    """
    try:
        # 저장된 파일의 경로와 ID 반환
        file_id = str(uuid.uuid4())
        file_path = await save_upload_file(file, f"{file_id}_{file.filename}", "user_images")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/clothing")
async def upload_clothing(
    file: UploadFile = File(...)
):
    """
    타겟 의류 이미지 업로드
    """
    try:
        # 저장된 파일의 경로와 ID 반환
        file_id = str(uuid.uuid4())
        file_path = await save_upload_file(file, f"{file_id}_{file.filename}", "target_clothing")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))