"""
인체 파싱 API 라우터
"""

import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import json
from datetime import datetime

from app.core.config import get_settings
from app.services.human_parsing.processor import get_human_parsing_processor

router = APIRouter(prefix="/human-parsing", tags=["Human Parsing"])
settings = get_settings()

class HumanParsingResponse(BaseModel):
    """인체 파싱 응답 모델"""
    file_id: str
    original_filename: str
    json_path: str
    class_count: int
    image_size: dict
    visualization_path: Optional[str] = None
    message: str = "이미지 처리 완료"
    success: bool = True

def save_to_local(json_data, collection_name, document_id):
    """로컬 파일 시스템에 JSON 데이터 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(settings.OUTPUT_DIR, 'firebase', collection_name)
    os.makedirs(output_dir, exist_ok=True)

    if document_id:
        output_file = os.path.join(output_dir, f"{document_id}.json")
    else:
        output_file = os.path.join(output_dir, f"doc_{timestamp}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] JSON 데이터를 {output_file}에 저장했습니다.")
    return output_file

@router.post("/process", response_model=HumanParsingResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    use_firebase: bool = Form(False),
    save_visualization: bool = Form(True),
    canvas_width: Optional[int] = Form(None),
    canvas_height: Optional[int] = Form(None)
):
    """
    이미지를 처리하여 인체 파싱 수행

    - **image**: 처리할 이미지 파일
    - **use_firebase**: Firebase에 결과 저장 여부
    - **save_visualization**: 시각화 이미지 저장 여부
    - **canvas_width**: 캔버스 너비 (None이면 원본 크기 사용)
    - **canvas_height**: 캔버스 높이 (None이면 원본 크기 사용)

    Returns:
        처리 결과와 JSON 파일 경로
    """
    try:
        # 임시 파일로 저장
        temp_dir = os.path.join(settings.BASE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(image.filename)[1]
        temp_file_path = os.path.join(temp_dir, f"{file_id}{file_extension}")

        # 업로드된 파일 저장
        with open(temp_file_path, "wb") as buffer:
            contents = await image.read()
            buffer.write(contents)

        # 출력 디렉토리 설정
        output_dir = os.path.join(settings.OUTPUT_DIR, "human_parsing")
        os.makedirs(output_dir, exist_ok=True)

        # 캔버스 크기 설정
        canvas_size = None
        if canvas_width is not None and canvas_height is not None:
            canvas_size = (canvas_width, canvas_height)  # (width, height)
            print(f"[INFO] Using custom canvas size: {canvas_width}x{canvas_height}")

        # 인체 파싱 프로세서 가져오기
        processor = get_human_parsing_processor()

        # 이미지 처리
        result_json, processed_image, segmentation_mask = processor.process_image(
            temp_file_path,
            output_dir=output_dir,
            save_json=True,
            canvas_size=canvas_size
        )

        # 결과 파일 경로
        file_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        json_path = os.path.join(output_dir, f"{file_name}_mask.json")
        vis_path = os.path.join(output_dir, f"{file_name}_visualization.png")

        # 클래스 수 계산
        class_count = len([c for c in np.unique(segmentation_mask) if c > 0])

        # Firebase 처리 (주석 처리 - 로컬 저장으로 대체)
        if use_firebase:
            # Firebase 대신 로컬에 저장
            save_path = save_to_local(result_json, "human_parsing", file_id)

        # 임시 파일 삭제 (백그라운드 작업으로)
        def cleanup_temp_file():
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        background_tasks.add_task(cleanup_temp_file)

        # 응답 데이터 준비
        response = {
            "file_id": file_id,
            "original_filename": image.filename,
            "json_path": f"/output/human_parsing/{file_name}_mask.json",
            "class_count": class_count,
            "image_size": result_json["image_size"],
            "visualization_path": f"/output/human_parsing/{file_name}_visualization.png" if save_visualization else None,
            "message": "이미지 처리 완료",
            "success": True
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{file_id}")
async def get_result(file_id: str):
    """처리 결과 JSON 파일 반환"""
    try:
        # JSON 파일 경로
        json_path = os.path.join(settings.OUTPUT_DIR, "human_parsing", f"{file_id}_mask.json")

        # 파일이 존재하는지 확인
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"Result not found: {file_id}")

        # JSON 파일 반환
        return FileResponse(json_path, media_type="application/json")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualization/{file_id}")
async def get_visualization(file_id: str):
    """시각화 이미지 파일 반환"""
    try:
        # 이미지 파일 경로
        image_path = os.path.join(settings.OUTPUT_DIR, "human_parsing", f"{file_id}_visualization.png")

        # 파일이 존재하는지 확인
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Visualization not found: {file_id}")

        # 이미지 파일 반환
        return FileResponse(image_path, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {"status": "healthy", "service": "human-parsing-api"}