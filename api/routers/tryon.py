"""
가상 피팅 관련 라우터
"""

import os
import sys
import uuid
import json
import shutil
from PIL import Image
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List
import traceback
from pydantic import BaseModel
import numpy as np

# detectron2 패치 적용
try:
    from api.patches.detectron2_patch import patch_detectron2
    patch_detectron2()
except:
    pass

from api.core.config import get_settings
from api.services.model_service import get_model_service
from api.services.densepose_service import DensePoseAPIClient  # apply_net 대신 DensePoseAPIClient 사용

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()
settings = get_settings()

# API 요청 모델
class TryOnRequest(BaseModel):
    user_image_id: str
    clothing_id: str
    category: str
    garment_description: Optional[str] = None
    parsing_id: Optional[str] = None
    pose_id: Optional[str] = None
    mask_id: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = None
    guidance_scale: Optional[float] = 2.0

@router.post("/process")
async def process_tryon(
    request: TryOnRequest
):
    """
    가상 피팅 처리

    Parameters:
    - user_image_id: 사용자 이미지 ID
    - clothing_id: 의류 이미지 ID
    - category: 의류 카테고리 (upper, lower, dress, outer)
    - garment_description: 의류 설명 (없으면 카테고리 기반 기본값 사용)
    - parsing_id: 인체 파싱 결과 ID (없으면 자동 파싱)
    - pose_id: 포즈 결과 ID (없으면 자동 생성)
    - mask_id: 마스크 이미지 ID (없으면 자동 생성)
    - num_inference_steps: 추론 단계 수 (기본값: 30)
    - seed: 랜덤 시드 (없으면 랜덤 사용)
    - guidance_scale: 가이던스 스케일 (기본값: 2.0)

    Returns:
    - 가상 피팅 결과
    """
    try:
        # 카테고리 검증
        if request.category not in settings.VALID_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 카테고리입니다. 유효한 값: {', '.join(settings.VALID_CATEGORIES)}"
            )

        # 1. 사용자 이미지 경로 확인
        user_img_dir = settings.USER_IMAGES_DIR
        user_img_files = [f for f in os.listdir(user_img_dir) if f.startswith(request.user_image_id)]

        if not user_img_files:
            raise HTTPException(status_code=404, detail=f"사용자 이미지를 찾을 수 없습니다: {request.user_image_id}")

        user_img_path = os.path.join(user_img_dir, user_img_files[0])

        # 2. 의류 이미지 경로 확인
        clothing_img_dir = settings.TARGET_CLOTHING_DIR
        clothing_img_files = [f for f in os.listdir(clothing_img_dir) if f.startswith(request.clothing_id)]

        if not clothing_img_files:
            raise HTTPException(status_code=404, detail=f"의류 이미지를 찾을 수 없습니다: {request.clothing_id}")

        clothing_img_path = os.path.join(clothing_img_dir, clothing_img_files[0])

        # 원본 이미지 크기 저장
        original_img = Image.open(user_img_path)
        original_size = original_img.size  # (width, height)

        # 3. 인체 파싱 결과 처리
        parsing_id = request.parsing_id
        parsing_result = None
        if not parsing_id:
            # 인체 파싱 요청
            print("인체 파싱 처리 시작...")
            from api.preprocess.humanparsing.run_parsing import Parsing
            parsing_model = Parsing(0)

            # 이미지 로드 및 처리
            img = Image.open(user_img_path).convert("RGB")
            img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

            # 인체 파싱 실행
            model_parse, _ = parsing_model(img_resized)

            # 파싱 결과를 JSON으로 변환
            parse_array = np.array(model_parse)
            parsing_result = {}

            # 각 클래스별 픽셀 좌표 저장
            for class_id in np.unique(parse_array):
                if class_id == 0:  # 배경 클래스 제외
                    continue

                # 해당 클래스의 픽셀 좌표 찾기
                y_coords, x_coords = np.where(parse_array == class_id)
                pixels = []

                for i in range(len(y_coords)):
                    pixels.append([int(x_coords[i]), int(y_coords[i])])

                parsing_result[str(class_id)] = pixels

            # 결과 저장
            parsing_id = str(uuid.uuid4())
            os.makedirs(settings.HUMAN_PARSING_DIR, exist_ok=True)
            parsing_json_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_mask.json")
            with open(parsing_json_path, "w") as f:
                json.dump(parsing_result, f)

            # 원본 크기 정보 저장
            size_info_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            print(f"인체 파싱 완료: {parsing_id}")
        else:
            # 기존 파싱 결과 로드
            parsing_json_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_mask.json")
            if not os.path.exists(parsing_json_path):
                raise HTTPException(status_code=404, detail=f"파싱 결과를 찾을 수 없습니다: {parsing_id}")

            with open(parsing_json_path, "r") as f:
                parsing_result = json.load(f)

            # 원본 크기 정보 확인
            size_info_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_size_info.json")
            if os.path.exists(size_info_path):
                with open(size_info_path, "r") as f:
                    size_info = json.load(f)
                    if "original_width" in size_info and "original_height" in size_info:
                        # 저장된 원본 크기와 현재 원본 크기 비교
                        if size_info["original_width"] != original_size[0] or size_info["original_height"] != original_size[1]:
                            print(f"경고: 저장된 원본 크기와 현재 이미지 크기가 다릅니다. 현재 이미지 크기를 사용합니다.")

        # 4. 포즈 추정 처리
        pose_id = request.pose_id
        pose_result = None
        if not pose_id:
            # 포즈 추정 요청
            print("포즈 추정 처리 시작...")
            from api.preprocess.openpose.run_openpose import OpenPose
            openpose_model = OpenPose(0)

            # 이미지 로드 및 처리
            img = Image.open(user_img_path).convert("RGB")
            img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

            # 포즈 추정 실행
            pose_result = openpose_model(img_resized)

            # 결과 저장
            pose_id = str(uuid.uuid4())
            os.makedirs(os.path.join(settings.OUTPUT_DIR, "pose"), exist_ok=True)
            pose_json_path = os.path.join(settings.OUTPUT_DIR, "pose", f"{pose_id}_keypoints.json")
            with open(pose_json_path, "w") as f:
                json.dump(pose_result, f)

            # 원본 크기 정보 저장
            size_info_path = os.path.join(settings.OUTPUT_DIR, "pose", f"{pose_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            print(f"포즈 추정 완료: {pose_id}")
        else:
            # 기존 포즈 결과 로드
            pose_json_path = os.path.join(settings.OUTPUT_DIR, "pose", f"{pose_id}_keypoints.json")
            if not os.path.exists(pose_json_path):
                raise HTTPException(status_code=404, detail=f"포즈 결과를 찾을 수 없습니다: {pose_id}")

            with open(pose_json_path, "r") as f:
                pose_result = json.load(f)

        # 5. DensePose 처리 (외부 API 사용)
        print("DensePose 처리 시작 (외부 API)...")
        # DensePose API 클라이언트 초기화
        densepose_client = DensePoseAPIClient(api_url=settings.DENSEPOSE_API_URL)

        # 이미지 로드
        img = Image.open(user_img_path).convert("RGB")
        img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

        # DensePose API 호출
        densepose_result = densepose_client.process_image(img_resized, {
            "include_body_parts": True,
            "include_uv_coordinates": True,
            "create_visualization": True,
            "output_directory": os.path.join(settings.OUTPUT_DIR, "densepose")
        })

        # API 응답 확인
        if "status" not in densepose_result or densepose_result["status"] != "success":
            error_msg = densepose_result.get("densepose_data", {}).get("error", "알 수 없는 오류")
            raise HTTPException(status_code=500, detail=f"DensePose API 오류: {error_msg}")

        # 원본 크기로 UV 좌표 스케일링
        scaled_result = densepose_client.scale_uv_coordinates(
            densepose_result,
            original_size=original_size
        )

        # 결과 저장
        densepose_id = str(uuid.uuid4())
        os.makedirs(os.path.join(settings.OUTPUT_DIR, "densepose"), exist_ok=True)
        densepose_json_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{densepose_id}_data.json")
        with open(densepose_json_path, "w") as f:
            json.dump(scaled_result.get("densepose_data", {}), f)

        # 시각화 결과 처리
        visualization_path = densepose_result.get("visualization_url")

        # 시각화 이미지가 없으면 기본 이미지 생성
        if not visualization_path:
            # 기본 이미지 생성 (빈 이미지 또는 로그 메시지)
            pose_pil = Image.new('RGB', (384, 512), (0, 0, 0))
            visualization_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{densepose_id}.png")
            pose_pil.save(visualization_path)
        elif os.path.isabs(visualization_path):
            # 절대 경로가 반환된 경우 새 경로로 복사
            new_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{densepose_id}.png")
            shutil.copy(visualization_path, new_path)
            visualization_path = new_path

        # 원본 크기로 시각화 이미지 리사이즈
        if os.path.exists(visualization_path) and original_size != (384, 512):
            try:
                vis_img = Image.open(visualization_path)
                resized_vis_img = vis_img.resize(original_size, Image.NEAREST)
                resized_vis_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{densepose_id}_original_size.png")
                resized_vis_img.save(resized_vis_path)

                # 원본 크기 이미지 사용
                densepose_img_path = resized_vis_path
            except Exception as e:
                print(f"경고: 시각화 이미지 리사이즈 실패: {str(e)}")
                densepose_img_path = visualization_path
        else:
            densepose_img_path = visualization_path

        # 원본 크기 정보 저장
        size_info_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{densepose_id}_size_info.json")
        with open(size_info_path, "w") as f:
            json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

        print(f"DensePose 처리 완료: {densepose_id} (경로: {densepose_img_path})")

        # 6. 마스크 생성
        mask_id = request.mask_id
        mask_path = None
        mask_gray_path = None

        if not mask_id:
            # 마스크 생성
            print("마스크 생성 시작...")
            from api.utils.utils_mask import get_mask_location

            # 마스크 생성
            mask_type = "hd"  # 기본 타입
            category_name = request.category
            if request.category == "upper":
                category_name = "upper_body"
            elif request.category == "lower":
                category_name = "lower_body"
            elif request.category == "dress":
                category_name = "dresses"

            # model_parse는 PIL.Image 타입이어야 함
            img = Image.open(user_img_path).convert("RGB")
            img_resized = img.resize((384, 512))

            from api.preprocess.humanparsing.run_parsing import Parsing
            parsing_model = Parsing(0)
            model_parse, _ = parsing_model(img_resized)

            mask, mask_gray = get_mask_location(mask_type, category_name, model_parse, pose_result)

            # 원본 크기로 리사이즈
            if original_size != (384, 512):
                mask = mask.resize(original_size, Image.NEAREST)
                mask_gray = mask_gray.resize(original_size, Image.NEAREST)

            # 결과 저장
            mask_id = str(uuid.uuid4())
            os.makedirs(os.path.join(settings.OUTPUT_DIR, "masks"), exist_ok=True)

            mask_path = os.path.join(settings.OUTPUT_DIR, "masks", f"{mask_id}_mask.png")
            mask_gray_path = os.path.join(settings.OUTPUT_DIR, "masks", f"{mask_id}_mask_gray.png")

            mask.save(mask_path)
            mask_gray.save(mask_gray_path)

            # 원본 크기 정보 저장
            size_info_path = os.path.join(settings.OUTPUT_DIR, "masks", f"{mask_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            print(f"마스크 생성 완료: {mask_id}")
        else:
            # 기존 마스크 파일 확인
            mask_path = os.path.join(settings.OUTPUT_DIR, "masks", f"{mask_id}_mask.png")
            mask_gray_path = os.path.join(settings.OUTPUT_DIR, "masks", f"{mask_id}_mask_gray.png")

            if not os.path.exists(mask_path):
                raise HTTPException(status_code=404, detail=f"마스크를 찾을 수 없습니다: {mask_id}")

            # 마스크 크기 확인 및 리사이즈
            mask = Image.open(mask_path)
            if mask.size != original_size:
                print(f"경고: 마스크 크기가 원본 이미지 크기와 다릅니다. 리사이즈합니다.")
                mask = mask.resize(original_size, Image.NEAREST)
                mask.save(mask_path)

                if os.path.exists(mask_gray_path):
                    mask_gray = Image.open(mask_gray_path)
                    mask_gray = mask_gray.resize(original_size, Image.NEAREST)
                    mask_gray.save(mask_gray_path)

        # 7. 의류 설명 확인
        garment_description = request.garment_description
        if not garment_description:
            # 카테고리별 기본 설명 사용
            if request.category == "upper":
                garment_description = "short sleeve t-shirt"
            elif request.category == "lower":
                garment_description = "jeans"
            elif request.category == "dress":
                garment_description = "casual dress"
            elif request.category == "outer":
                garment_description = "jacket"

        # 8. 가상 피팅 처리
        try:
            print("가상 피팅 처리 시작...")
            model_service = get_model_service()

            # 결과 이미지 경로
            result_id = str(uuid.uuid4())
            output_path = os.path.join(settings.TRYON_DIR, f"{result_id}.jpg")

            # 가상 피팅 실행
            result = model_service.generate_tryon_image(
                person_image_path=user_img_path,
                garment_image_path=clothing_img_path,
                mask_image_path=mask_path,
                pose_image_path=densepose_img_path,
                garment_description=garment_description,
                category=request.category,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                guidance_scale=request.guidance_scale,
                output_path=output_path
            )

            if result["status"] != "success":
                raise HTTPException(status_code=500, detail=result["message"])

            print(f"가상 피팅 완료: {result_id}")

            # 응답 데이터 구성
            response_data = {
                "status": "success",
                "result_id": result_id,
                "category": request.category,
                "image_url": f"/output/tryon/{result_id}.jpg",
                "original_size": {
                    "width": original_size[0],
                    "height": original_size[1]
                },
                "input": {
                    "user_image_id": request.user_image_id,
                    "clothing_id": request.clothing_id,
                    "parsing_id": parsing_id,
                    "pose_id": pose_id,
                    "densepose_id": densepose_id,
                    "mask_id": mask_id,
                    "garment_description": garment_description
                },
                "message": "가상 피팅 처리 완료"
            }
            
            return response_data
            
        except Exception as e:
            print(f"가상 피팅 처리 중 오류: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"가상 피팅 처리 중 오류: {str(e)}")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"처리 중 예외 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"가상 피팅 처리 중 오류: {str(e)}")

@router.get("/result/{result_id}")
async def get_tryon_result(result_id: str):
    """
    가상 피팅 결과 조회
    
    Parameters:
    - result_id: 결과 ID
    
    Returns:
    - 가상 피팅 결과 이미지
    """
    try:
        # 결과 이미지 경로 확인
        result_path = os.path.join(settings.TRYON_DIR, f"{result_id}.jpg")
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail=f"결과 이미지를 찾을 수 없습니다: {result_id}")
        
        return FileResponse(result_path, media_type="image/jpeg")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 중 오류: {str(e)}")

@router.get("/status")
async def get_model_status():
    """
    모델 상태 조회
    
    Returns:
    - 모델 상태 정보
    """
    try:
        model_service = get_model_service()
        return model_service.get_device_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 상태 조회 중 오류: {str(e)}")