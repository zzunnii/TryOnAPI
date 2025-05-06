"""
의류 이미지 관련 라우터
"""

import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from typing import Optional, List

from api.core.config import get_settings
from api.utils.image_utils import save_image

router = APIRouter()
settings = get_settings()

@router.post("/upload")
async def upload_clothing(
    file: UploadFile = File(...),
    category: str = Form(None)
):
    """
    의류 이미지 업로드
    
    Parameters:
    - file: 업로드할 의류 이미지 파일
    - category: 의류 카테고리 (upper, lower, dress, outer)
    
    Returns:
    - clothing_id: 의류 이미지 ID
    - filename: 원본 파일명
    - file_path: 저장된 파일 경로
    - category: 의류 카테고리
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다. JPG 또는 PNG만 지원합니다.")
        
        # 카테고리 검증
        if category and category not in settings.VALID_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 카테고리입니다. 유효한 값: {', '.join(settings.VALID_CATEGORIES)}"
            )
        
        # 기본 카테고리 설정
        if not category:
            category = "upper"  # 기본값 설정
        
        # 저장 디렉토리 확인 및 생성
        clothing_id = str(uuid.uuid4())
        save_dir = settings.TARGET_CLOTHING_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일 저장 경로
        file_path = os.path.join(save_dir, f"{clothing_id}_{category}{file_ext}")
        
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "clothing_id": clothing_id,
            "filename": file.filename,
            "file_path": file_path,
            "category": category
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"의류 이미지 업로드 중 오류: {str(e)}")

@router.get("/{clothing_id}")
async def get_clothing(clothing_id: str):
    """
    의류 이미지 조회
    
    Parameters:
    - clothing_id: 의류 이미지 ID
    
    Returns:
    - 의류 이미지 파일
    """
    try:
        # 의류 이미지 파일 경로 탐색
        file_dir = settings.TARGET_CLOTHING_DIR
        matching_files = [f for f in os.listdir(file_dir) if f.startswith(clothing_id)]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail=f"의류 이미지 ID를 찾을 수 없습니다: {clothing_id}")
        
        file_path = os.path.join(file_dir, matching_files[0])
        
        return FileResponse(file_path, media_type="image/jpeg")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"의류 이미지 조회 중 오류: {str(e)}")

@router.delete("/{clothing_id}")
async def delete_clothing(clothing_id: str):
    """
    의류 이미지 삭제
    
    Parameters:
    - clothing_id: 의류 이미지 ID
    
    Returns:
    - 삭제 상태
    """
    try:
        # 의류 이미지 파일 경로 탐색
        file_dir = settings.TARGET_CLOTHING_DIR
        matching_files = [f for f in os.listdir(file_dir) if f.startswith(clothing_id)]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail=f"의류 이미지 ID를 찾을 수 없습니다: {clothing_id}")
        
        # 파일 삭제
        for file_name in matching_files:
            file_path = os.path.join(file_dir, file_name)
            os.remove(file_path)
        
        return {
            "status": "success",
            "message": f"의류 이미지 ID {clothing_id}가 삭제되었습니다.",
            "deleted_files": matching_files
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"의류 이미지 삭제 중 오류: {str(e)}")

@router.get("/categories")
async def get_clothing_categories():
    """
    지원하는 의류 카테고리 목록 조회
    
    Returns:
    - 의류 카테고리 목록
    """
    return {
        "categories": settings.VALID_CATEGORIES,
        "description": {
            "upper": "상의 (셔츠, 티셔츠, 블라우스 등)",
            "lower": "하의 (바지, 스커트 등)",
            "dress": "원피스",
            "outer": "아우터 (자켓, 코트 등)"
        }
    }