"""
전처리 관련 라우터 (인체 파싱, DensePose 등)
"""

import os
import sys
import uuid
import json
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List
import traceback

# detectron2 패치 적용
try:
    from api.patches.detectron2_patch import patch_detectron2
    patch_detectron2()
except:
    pass

from api.core.config import get_settings
from api.utils.image_utils import save_image
from api.services.densepose_service import DensePoseAPIClient

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()
settings = get_settings()

# DensePose API 클라이언트 인스턴스 생성
densepose_client = DensePoseAPIClient(api_url=settings.DENSEPOSE_API_URL)

@router.post("/human-parsing")
async def process_human_parsing(
    image: UploadFile = File(...),
    save_visualization: bool = Form(False)
):
    """
    인체 파싱 처리

    Parameters:
    - image: 인체 이미지 파일
    - save_visualization: 시각화 이미지 저장 여부

    Returns:
    - 인체 파싱 결과
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(image.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다. JPG 또는 PNG만 지원합니다.")

        # 이미지 저장
        result_id = str(uuid.uuid4())
        temp_img_path = os.path.join(settings.TEMP_DIR, f"parsing_temp_{result_id}{file_ext}")

        with open(temp_img_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # 인체 파싱 모델 가져오기
        try:
            from api.preprocess.humanparsing.run_parsing import Parsing
            parsing_model = Parsing(0)  # 0: CPU, 1: GPU
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="인체 파싱 모델을 로드할 수 없습니다. preprocess.humanparsing 모듈을 확인하세요."
            )

        # 이미지 로드 및 처리
        img = Image.open(temp_img_path).convert("RGB")
        original_size = img.size
        img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

        # 인체 파싱 실행
        try:
            model_parse, _ = parsing_model(img_resized)

            # 결과 저장 디렉토리 확인
            os.makedirs(settings.HUMAN_PARSING_DIR, exist_ok=True)

            # 파싱 결과를 JSON으로 변환
            parse_array = np.array(model_parse)
            parse_result = {}

            # 각 클래스별 픽셀 좌표 저장
            for class_id in np.unique(parse_array):
                if class_id == 0:  # 배경 클래스 제외
                    continue

                # 해당 클래스의 픽셀 좌표 찾기
                y_coords, x_coords = np.where(parse_array == class_id)
                pixels = []

                for i in range(len(y_coords)):
                    pixels.append([int(x_coords[i]), int(y_coords[i])])

                parse_result[str(class_id)] = pixels

            # 결과 저장
            result_json_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{result_id}_mask.json")
            with open(result_json_path, "w") as f:
                json.dump(parse_result, f)

            # 시각화 이미지 생성 (선택적)
            visualization_path = None
            if save_visualization:
                from api.utils.image_utils import create_mask_from_segmentation

                # 전체 인체 마스크 생성
                all_parts = list(range(1, 20))  # 클래스 1-19까지 사용
                visualization = create_mask_from_segmentation(parse_result, all_parts, (384, 512))

                # 시각화 이미지 저장
                visualization_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{result_id}_visualization.png")
                visualization.save(visualization_path)

                # 컬러 시각화 추가
                color_vis = np.zeros((512, 384, 3), dtype=np.uint8)
                colors = {
                    1: (255, 0, 0),    # 모자
                    2: (0, 255, 0),    # 머리카락
                    3: (0, 0, 255),    # 선글라스
                    4: (255, 255, 0),  # 상의
                    5: (255, 0, 255),  # 치마
                    6: (0, 255, 255),  # 바지
                    7: (128, 0, 0),    # 드레스
                    8: (0, 128, 0),    # 벨트
                    9: (0, 0, 128),    # 왼쪽 신발
                    10: (128, 128, 0), # 오른쪽 신발
                    11: (128, 0, 128), # 얼굴
                    12: (0, 128, 128), # 왼쪽 다리
                    13: (128, 128, 128), # 오른쪽 다리
                    14: (64, 0, 0),    # 왼쪽 팔
                    15: (0, 64, 0),    # 오른쪽 팔
                    16: (0, 0, 64),    # 가방
                    17: (64, 64, 0),   # 스카프
                }

                for class_id, color in colors.items():
                    if str(class_id) in parse_result:
                        for x, y in parse_result[str(class_id)]:
                            if 0 <= x < 384 and 0 <= y < 512:
                                color_vis[y, x] = color

                comparison_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{result_id}_comparison.png")
                Image.fromarray(color_vis).save(comparison_path)

            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            # 원본 크기 저장
            size_info_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{result_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            # 응답 데이터 구성
            response_data = {
                "status": "success",
                "file_id": result_id,
                "mask_json_path": result_json_path,
                "class_counts": {str(class_id): len(pixels) for class_id, pixels in parse_result.items()},
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "message": "인체 파싱 처리 완료"
            }

            if visualization_path:
                response_data["visualization_path"] = f"/output/human_parsing/{result_id}_visualization.png"
                response_data["comparison_path"] = f"/output/human_parsing/{result_id}_comparison.png"

            return response_data

        except Exception as e:
            print(f"인체 파싱 처리 중 오류: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"인체 파싱 처리 중 오류: {str(e)}")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"처리 중 예외 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"인체 파싱 처리 중 오류: {str(e)}")

@router.post("/densepose")
async def process_densepose(
    image: UploadFile = File(...),
    save_visualization: bool = Form(False)
):
    """
    DensePose 처리 (외부 API 활용)

    Parameters:
    - image: 인체 이미지 파일
    - save_visualization: 시각화 이미지 저장 여부

    Returns:
    - DensePose 처리 결과
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(image.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다. JPG 또는 PNG만 지원합니다.")

        # 이미지 저장
        result_id = str(uuid.uuid4())
        temp_img_path = os.path.join(settings.TEMP_DIR, f"densepose_temp_{result_id}{file_ext}")

        with open(temp_img_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # 이미지 로드
        img = Image.open(temp_img_path).convert("RGB")
        original_size = img.size
        img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

        # DensePose 처리 실행 (외부 API 호출)
        try:
            # DensePose API 클라이언트를 사용하여 처리
            densepose_result = densepose_client.process_image(img_resized, {
                "include_body_parts": True,
                "include_uv_coordinates": True,
                "sample_uv": None,  # 모든 UV 좌표 가져오기
                "create_visualization": save_visualization,
                "output_directory": settings.OUTPUT_DIR
            })

            # API 응답에서 오류 확인
            if "error" in densepose_result.get("densepose_data", {}):
                raise ValueError(densepose_result["densepose_data"]["error"])

            # 원본 크기로 UV 좌표 스케일링
            scaled_result = densepose_client.scale_uv_coordinates(
                densepose_result,
                original_size=original_size
            )

            # 결과 저장 디렉토리 확인
            os.makedirs(os.path.join(settings.OUTPUT_DIR, "densepose"), exist_ok=True)

            # 결과 저장
            result_json_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{result_id}_densepose.json")
            with open(result_json_path, "w") as f:
                json.dump(scaled_result.get("densepose_data", {}), f)

            # 원본 크기 저장
            size_info_path = os.path.join(settings.OUTPUT_DIR, "densepose", f"{result_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            # 시각화 이미지 처리
            visualization_path = scaled_result.get("visualization_url")
            if visualization_path:
                # 원본 경로가 절대 경로인 경우 상대 경로로 변환
                if os.path.isabs(visualization_path):
                    visualization_path = os.path.relpath(
                        visualization_path,
                        start=os.path.dirname(settings.OUTPUT_DIR)
                    )

            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            # 응답 데이터 생성
            response_data = {
                "status": "success",
                "file_id": result_id,
                "densepose_json_path": result_json_path,
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "message": "DensePose 처리 완료 (외부 API 활용)"
            }

            if visualization_path:
                response_data["visualization_path"] = visualization_path

            return response_data

        except Exception as e:
            print(f"DensePose 처리 중 오류: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"DensePose 처리 중 오류: {str(e)}")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"처리 중 예외 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DensePose 처리 중 오류: {str(e)}")

@router.post("/pose-estimation")
async def process_pose_estimation(
    image: UploadFile = File(...),
    save_visualization: bool = Form(False)
):
    """
    포즈 추정 처리 (OpenPose)

    Parameters:
    - image: 인체 이미지 파일
    - save_visualization: 시각화 이미지 저장 여부

    Returns:
    - 포즈 추정 결과
    """
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(image.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="허용되지 않는 파일 형식입니다. JPG 또는 PNG만 지원합니다.")

        # 이미지 저장
        result_id = str(uuid.uuid4())
        temp_img_path = os.path.join(settings.TEMP_DIR, f"pose_temp_{result_id}{file_ext}")

        with open(temp_img_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # OpenPose 모델 가져오기
        try:
            from api.preprocess.openpose.run_openpose import OpenPose
            openpose_model = OpenPose(0)  # 0: CPU, 1: GPU
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="OpenPose 모델을 로드할 수 없습니다. preprocess.openpose 모듈을 확인하세요."
            )

        # 이미지 로드 및 처리
        img = Image.open(temp_img_path).convert("RGB")
        original_size = img.size
        img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

        # 포즈 추정 실행
        try:
            keypoints = openpose_model(img_resized)

            # 결과 저장 디렉토리 확인
            output_dir = os.path.join(settings.OUTPUT_DIR, "pose")
            os.makedirs(output_dir, exist_ok=True)

            # 결과 저장
            result_json_path = os.path.join(output_dir, f"{result_id}_keypoints.json")
            with open(result_json_path, "w") as f:
                json.dump(keypoints, f)

            # 원본 크기 저장
            size_info_path = os.path.join(output_dir, f"{result_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            # 시각화 이미지 생성 (선택적)
            visualization_path = None
            if save_visualization:
                # 시각화 함수 (간단한 구현)
                def visualize_keypoints(image, keypoints):
                    vis_img = image.copy()
                    draw = ImageDraw.Draw(vis_img)

                    # 관절 점 그리기
                    for kp in keypoints["pose_keypoints_2d"]:
                        if len(kp) >= 3 and kp[2] > 0:  # 신뢰도 확인
                            x, y = int(kp[0]), int(kp[1])
                            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0))

                    return vis_img

                from PIL import ImageDraw
                visualization = visualize_keypoints(img_resized, keypoints)

                # 시각화 이미지 저장
                visualization_path = os.path.join(output_dir, f"{result_id}_visualization.png")
                visualization.save(visualization_path)

            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            # 응답 데이터 구성
            response_data = {
                "status": "success",
                "file_id": result_id,
                "keypoints_json_path": result_json_path,
                "keypoints": keypoints,
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "message": "포즈 추정 처리 완료"
            }

            if visualization_path:
                response_data["visualization_path"] = f"/output/pose/{result_id}_visualization.png"

            return response_data

        except Exception as e:
            print(f"포즈 추정 처리 중 오류: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"포즈 추정 처리 중 오류: {str(e)}")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"처리 중 예외 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"포즈 추정 처리 중 오류: {str(e)}")

@router.post("/mask-generation")
async def generate_mask(
    image: UploadFile = File(...),
    category: str = Form(...),
    parsing_id: Optional[str] = Form(None),
    pose_id: Optional[str] = Form(None)
):
    """
    의류 카테고리별 마스크 생성

    Parameters:
    - image: 인체 이미지 파일
    - category: 의류 카테고리 (upper, lower, dress, outer)
    - parsing_id: 인체 파싱 결과 ID (없으면 자동 파싱)
    - pose_id: 포즈 추정 결과 ID (없으면 자동 추정)

    Returns:
    - 마스크 생성 결과
    """
    try:
        # 카테고리 검증
        if category not in settings.VALID_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 카테고리입니다. 유효한 값: {', '.join(settings.VALID_CATEGORIES)}"
            )

        # 이미지 저장
        result_id = str(uuid.uuid4())
        file_ext = os.path.splitext(image.filename)[1].lower()
        temp_img_path = os.path.join(settings.TEMP_DIR, f"mask_temp_{result_id}{file_ext}")

        with open(temp_img_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # 이미지 로드
        img = Image.open(temp_img_path).convert("RGB")
        original_size = img.size
        img_resized = img.resize((384, 512))  # 모델 입력 크기로 리사이즈

        # 1. 인체 파싱 실행 (ID가 없는 경우)
        parsing_result = None
        model_parse = None
        if not parsing_id:
            # 인체 파싱 모델 호출
            from api.preprocess.humanparsing.run_parsing import Parsing
            parsing_model = Parsing(0)
            model_parse, _ = parsing_model(img_resized)

            # 파싱 결과 변환
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

            # 파싱 결과 저장
            os.makedirs(settings.HUMAN_PARSING_DIR, exist_ok=True)
            parsing_json_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{result_id}_mask.json")
            with open(parsing_json_path, "w") as f:
                json.dump(parsing_result, f)

            parsing_id = result_id
        else:
            # 기존 파싱 결과 로드
            parsing_json_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_mask.json")
            if not os.path.exists(parsing_json_path):
                raise HTTPException(status_code=404, detail=f"파싱 결과를 찾을 수 없습니다: {parsing_id}")

            with open(parsing_json_path, "r") as f:
                parsing_result = json.load(f)

            # 파싱 결과로부터 model_parse 이미지 생성
            model_parse = Image.new('L', (384, 512), 0)
            for class_id, pixels in parsing_result.items():
                class_id_int = int(class_id)
                for x, y in pixels:
                    if 0 <= x < 384 and 0 <= y < 512:
                        model_parse.putpixel((x, y), class_id_int)

            # 파싱 결과의 원본 크기 확인
            size_info_path = os.path.join(settings.HUMAN_PARSING_DIR, f"{parsing_id}_size_info.json")
            if os.path.exists(size_info_path):
                with open(size_info_path, "r") as f:
                    size_info = json.load(f)
                    if "original_width" in size_info and "original_height" in size_info:
                        original_size = (size_info["original_width"], size_info["original_height"])

        # 2. 포즈 추정 실행 (ID가 없는 경우)
        pose_result = None
        if not pose_id:
            # OpenPose 모델 호출
            from api.preprocess.openpose.run_openpose import OpenPose
            openpose_model = OpenPose(0)
            pose_result = openpose_model(img_resized)

            # 포즈 결과 저장
            os.makedirs(os.path.join(settings.OUTPUT_DIR, "pose"), exist_ok=True)
            pose_json_path = os.path.join(settings.OUTPUT_DIR, "pose", f"{result_id}_keypoints.json")
            with open(pose_json_path, "w") as f:
                json.dump(pose_result, f)

            pose_id = result_id
        else:
            # 기존 포즈 결과 로드
            pose_json_path = os.path.join(settings.OUTPUT_DIR, "pose", f"{pose_id}_keypoints.json")
            if not os.path.exists(pose_json_path):
                raise HTTPException(status_code=404, detail=f"포즈 결과를 찾을 수 없습니다: {pose_id}")

            with open(pose_json_path, "r") as f:
                pose_result = json.load(f)

        # 3. 카테고리별 마스크 생성
        try:
            # utils_mask.py 사용
            from api.utils.utils_mask import get_mask_location

            # 마스크 생성
            mask_type = "hd"  # 기본 타입
            category_name = category
            if category == "upper":
                category_name = "upper_body"
            elif category == "lower":
                category_name = "lower_body"
            elif category == "dress":
                category_name = "dresses"

            mask, mask_gray = get_mask_location(mask_type, category_name, model_parse, pose_result)

            # 원본 크기로 복원
            if original_size != (384, 512):
                mask = mask.resize(original_size, Image.NEAREST)
                mask_gray = mask_gray.resize(original_size, Image.NEAREST)

            # 결과 저장 디렉토리 확인
            output_dir = os.path.join(settings.OUTPUT_DIR, "masks")
            os.makedirs(output_dir, exist_ok=True)

            # 마스크 이미지 저장
            mask_path = os.path.join(output_dir, f"{result_id}_mask.png")
            mask_gray_path = os.path.join(output_dir, f"{result_id}_mask_gray.png")

            mask.save(mask_path)
            mask_gray.save(mask_gray_path)

            # 원본 크기 저장
            size_info_path = os.path.join(output_dir, f"{result_id}_size_info.json")
            with open(size_info_path, "w") as f:
                json.dump({"original_width": original_size[0], "original_height": original_size[1]}, f)

            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            # 응답 데이터 구성
            response_data = {
                "status": "success",
                "file_id": result_id,
                "category": category,
                "mask_path": f"/output/masks/{result_id}_mask.png",
                "mask_gray_path": f"/output/masks/{result_id}_mask_gray.png",
                "parsing_id": parsing_id,
                "pose_id": pose_id,
                "original_size": {"width": original_size[0], "height": original_size[1]},
                "message": f"{category} 카테고리 마스크 생성 완료"
            }
            
            return response_data
            
        except Exception as e:
            print(f"마스크 생성 중 오류: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"마스크 생성 중 오류: {str(e)}")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"처리 중 예외 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"마스크 생성 중 오류: {str(e)}")