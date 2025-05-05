import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List
import os
import uuid
import json
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.segmentation_service import segment_image, remove_background, create_agnostic_image

router = APIRouter()
settings = get_settings()


# API 응답 모델 정의 (Pydantic 모델로 변경)
class ImageResponse(BaseModel):
    id: str
    original_id: str
    segmentation_path: Optional[str] = None
    mask_path: Optional[str] = None
    json_path: Optional[str] = None
    image_path: Optional[str] = None
    message: str = "Operation completed successfully"


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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/json/{result_id}")
async def get_segmentation_json(result_id: str):
    """
    세그멘테이션 결과 JSON 반환
    """
    try:
        # JSON 파일 경로 찾기
        json_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{result_id}_mask.json")

        # 파일이 존재하는지 확인
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"JSON result with ID {result_id} not found")

        # 파일 반환
        return FileResponse(json_path, media_type="application/json")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=ImageResponse)
async def segment_image_endpoint(
        image_id: str,
        canvas_width: Optional[int] = Form(None),
        canvas_height: Optional[int] = Form(None)
):
    """
    이미지 세그멘테이션 수행 (Class 20)

    - **image_id**: 처리할 이미지 ID
    - **canvas_width**: 캔버스 너비 (None이면 원본 크기 사용)
    - **canvas_height**: 캔버스 높이 (None이면 원본 크기 사용)
    """
    try:
        # 이미지 ID로 저장된 이미지 경로 찾기
        image_path = os.path.join(settings.OUTPUT_DIR, "uploads", f"{image_id}.jpg")

        # 이미지가 존재하는지 확인
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")

        # 캔버스 크기 설정
        canvas_size = None
        if canvas_width is not None and canvas_height is not None:
            canvas_size = (canvas_width, canvas_height)  # (width, height)
            print(f"[INFO] Using custom canvas size: {canvas_width}x{canvas_height}")

        # 인체 파싱 프로세서 가져오기
        from app.services.human_parsing.processor import get_human_parsing_processor
        processor = get_human_parsing_processor()

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 세그멘테이션 처리 및 원본 크기로 변환
        processed_image, pred_mask, orig_mask, overlay_original = processor.process_and_segment(
            image_rgb,
            canvas_size=canvas_size
        )

        if orig_mask is None:
            print("[WARN] 이미지에서 사람을 찾을 수 없습니다.")
            # 빈 결과 반환
            result = {
                'segmentation_path': image_path,
                'mask_path': os.path.join(settings.OUTPUT_DIR, "segmentation", f"{image_id}_mask.png"),
                'json_path': os.path.join(settings.OUTPUT_DIR, "segmentation", f"{image_id}_mask.json")
            }

            # 빈 JSON 생성
            h, w = image.shape[:2]
            empty_json = {
                'image_size': {'width': w, 'height': h},
                'class_masks': {}
            }

            with open(result['json_path'], 'w', encoding='utf-8') as f:
                json.dump(empty_json, f, ensure_ascii=False, indent=2)

        else:
            # 결과 저장
            segmentation_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{image_id}_segmentation.jpg")
            mask_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{image_id}_mask.png")
            json_path = os.path.join(settings.OUTPUT_DIR, "segmentation", f"{image_id}_mask.json")

            cv2.imwrite(segmentation_path, cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, orig_mask)

            # JSON 마스크 생성 및 저장
            h, w = image.shape[:2]
            mask_json = processor.mask_to_json(orig_mask, (w, h))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(mask_json, f, ensure_ascii=False, indent=2)

            result = {
                'segmentation_path': segmentation_path,
                'mask_path': mask_path,
                'json_path': json_path
            }

        # 세그멘테이션 결과에서 파일 이름만 추출
        segmentation_filename = os.path.basename(result['segmentation_path'])
        mask_filename = os.path.basename(result['mask_path'])
        json_filename = os.path.basename(result['json_path'])

        # 결과 반환 (Pydantic 모델 사용)
        return ImageResponse(
            id=str(uuid.uuid4()),
            original_id=image_id,
            segmentation_path=f"/output/segmentation/{segmentation_filename}",
            mask_path=f"/output/segmentation/{mask_filename}",
            json_path=f"/output/segmentation/{json_filename}",
            message="Segmentation completed successfully"
        )
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
        return ImageResponse(
            id=str(uuid.uuid4()),
            original_id=segmentation_id,
            image_path=f"/output/no_background/{result_filename}",
            message="Background removal completed successfully"
        )
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
        return ImageResponse(
            id=str(uuid.uuid4()),
            original_id=segmentation_id,
            image_path=f"/output/agnostic/{result_filename}",
            message="Agnostic image created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))