"""
MediaPipe 라우터 - 키포인트 추출 API
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import os
import uuid
import json

from app.services.mediapipe_service import get_mediapipe_service
from app.services.image_service import save_upload_file
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

@router.post("/keypoints")
async def get_keypoints(
    file: UploadFile = File(...)
):
    """
    MediaPipe를 사용하여 인체 키포인트 추출
    
    - **file**: 분석할 이미지 파일
    
    Returns:
        키포인트 정보와 메타데이터
    """
    try:
        # 서비스 인스턴스 가져오기
        mediapipe_service = get_mediapipe_service()
        
        # 파일 임시 저장
        temp_dir = os.path.join(settings.TEMP_DIR, "mediapipe")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(temp_dir, f"{file_id}{file_extension}")
        
        # 업로드된 파일 저장
        with open(temp_file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # MediaPipe 처리
        keypoints_data = mediapipe_service.extract_keypoints(temp_file_path)
        
        # 출력 디렉토리 설정
        output_dir = os.path.join(settings.OUTPUT_DIR, "mediapipe")
        os.makedirs(output_dir, exist_ok=True)
        
        # 결과 JSON 파일 저장
        json_path = os.path.join(output_dir, f"{file_id}_keypoints.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data, f, ensure_ascii=False, indent=2)
        
        # 시각화 이미지 저장 (선택적)
        vis_path = os.path.join(output_dir, f"{file_id}_visualization.png")
        mediapipe_service.visualize_keypoints(temp_file_path, keypoints_data, vis_path)
        
        # 응답 데이터
        response = {
            "file_id": file_id,
            "original_filename": file.filename,
            "keypoints_count": len(keypoints_data["pose_landmarks"]) if "pose_landmarks" in keypoints_data else 0,
            "json_path": f"/output/mediapipe/{file_id}_keypoints.json",
            "visualization_path": f"/output/mediapipe/{file_id}_visualization.png",
            "message": "MediaPipe 키포인트 추출 완료",
            "success": True
        }
        
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {"status": "healthy", "service": "mediapipe-api"}
