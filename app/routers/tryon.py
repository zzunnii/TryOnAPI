import os
import uuid
import json
import aiohttp
import asyncio
import numpy as np
import cv2
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict, Any, List

from app.services.image_service import save_upload_file
from app.services.agnostic_service import get_agnostic_service
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

class IDMVTONRequest(BaseModel):
    user_image_id: str
    clothing_id: str
    parsing_id: str
    category: str
    densepose_data: str
    agnostic_id: Optional[str] = None  # 선택적 agnostic 이미지 ID

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

async def call_densepose_api(image_path):
    """
    Detectron2 DensePose API 호출
    """
    try:
        # Detectron2 DensePose API URL
        DENSEPOSE_API_URL = "http://localhost:8001/process/file"
        
        # 이미지 파일 확인
        if not os.path.exists(image_path):
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        # 이미지 파일 업로드
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as img_file:
                data = aiohttp.FormData()
                data.add_field('file', img_file, 
                              filename=os.path.basename(image_path),
                              content_type='image/jpeg')
                
                # 파라미터 추가
                data.add_field('include_body_parts', 'true')
                data.add_field('include_uv_coordinates', 'true')
                data.add_field('create_visualization', 'false')
                
                # API 요청 실행
                try:
                    async with session.post(DENSEPOSE_API_URL, data=data) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            response_text = await response.text()
                            print(f"DensePose API 호출 오류: {response.status}, {response_text}")
                except aiohttp.ClientError as e:
                    print(f"DensePose API 서버 연결 오류: {str(e)}")
                    print("Detectron2 API 서버가 실행 중인지 확인해주세요.")
        
        # API 호출 실패 시 목업 응답 사용
        print("실제 API 호출 실패 - 목업 응답 사용 (테스트용)")
        await asyncio.sleep(0.5)  # API 호출 시뮬레이션
        
        # 목업 응답 생성
        # 이미지 크기 가져오기
        img = cv2.imread(image_path)
        height, width = img.shape[:2] if img is not None else (1024, 768)
        
        # 24개 바디 파트에 대한 세그먼트 정보
        segments = {}
        for i in range(1, 25):
            # 각 파트마다 임의의 픽셀 좌표 생성
            num_pixels = np.random.randint(50, 200)
            pixels = []
            for _ in range(num_pixels):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                pixels.append([int(x), int(y)])
            
            segments[str(i)] = pixels
        
        # UV 좌표 생성
        uv_coordinates = []
        num_points = 500  # UV 좌표 수
        for _ in range(num_points):
            part_id = np.random.randint(1, 25)
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            uv_coordinates.append({
                "part_id": part_id,
                "u": float(u),
                "v": float(v),
                "x": int(x),
                "y": int(y)
            })
        
        mock_data = {
            "status": "success",
            "densepose_data": {
                "segments": segments,
                "uv_coordinates": uv_coordinates,
                "image_width": width,
                "image_height": height,
                "num_persons": 1,
                "is_mock_data": True  # 가짜 데이터 표시
            }
        }
        
        return mock_data
        
    except Exception as e:
        print(f"DensePose API 호출 오류: {str(e)}")
        return {"status": "error", "message": str(e)}

@router.post("/idm-vton")
async def idm_vton_process(
    request_data: IDMVTONRequest
):
    """
    IDM VTON 모델을 사용한 가상 착용 처리
    
    - **user_image_id**: 사용자 이미지 ID
    - **clothing_id**: 의류 이미지 ID
    - **parsing_id**: 인체 파싱 결과 ID
    - **category**: 의류 카테고리 (upper, lower, dress, outer)
    - **densepose_data**: DensePose 분석 데이터 (JSON 문자열)
    """
    try:
        # 1. 사용자 이미지 경로 확인
        user_img_dir = os.path.join(settings.TEMP_DIR, "user_images")
        user_img_files = [f for f in os.listdir(user_img_dir) if f.startswith(request_data.user_image_id)]
        
        if not user_img_files:
            raise HTTPException(status_code=404, detail=f"사용자 이미지를 찾을 수 없습니다: {request_data.user_image_id}")
        
        user_img_path = os.path.join(user_img_dir, user_img_files[0])
        
        # 2. 의류 이미지 경로 확인
        clothing_img_dir = os.path.join(settings.TEMP_DIR, "target_clothing")
        clothing_img_files = [f for f in os.listdir(clothing_img_dir) if f.startswith(request_data.clothing_id)]
        
        if not clothing_img_files:
            raise HTTPException(status_code=404, detail=f"의류 이미지를 찾을 수 없습니다: {request_data.clothing_id}")
        
        clothing_img_path = os.path.join(clothing_img_dir, clothing_img_files[0])
        
        # 3. 인체 파싱 결과 확인
        parsing_result_path = os.path.join(settings.OUTPUT_DIR, "human_parsing", f"{request_data.parsing_id}_mask.json")
        
        if not os.path.exists(parsing_result_path):
            raise HTTPException(status_code=404, detail=f"인체 파싱 결과를 찾을 수 없습니다: {request_data.parsing_id}")
        
        # 4. DensePose 데이터 파싱
        try:
            densepose_data = json.loads(request_data.densepose_data)
            
            # 미디어파이프 키포인트가 있으면 함께 로드
            mediapipe_keypoints = None
            mediapipe_path = os.path.join(settings.OUTPUT_DIR, "mediapipe", f"{request_data.user_image_id}_keypoints.json")
            
            if os.path.exists(mediapipe_path):
                try:
                    with open(mediapipe_path, 'r') as f:
                        mediapipe_data = json.load(f)
                        mediapipe_keypoints = mediapipe_data.get("pose_landmarks", [])
                        print(f"미디어파이프 키포인트 로드 완료: {len(mediapipe_keypoints)} 개")
                        
                        # DensePose 데이터에 키포인트 정보 추가
                        densepose_data["keypoints"] = mediapipe_keypoints
                except Exception as e:
                    print(f"미디어파이프 키포인트 로드 오류 (무시됨): {str(e)}")
                    
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="DensePose 데이터 형식이 잘못되었습니다")
        
        # 5. 카테고리 검증
        valid_categories = ["upper", "lower", "dress", "outer"]
        if request_data.category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 카테고리입니다. 유효한 값: {', '.join(valid_categories)}"
            )
            
        # 6. 결과 이미지 저장 경로 설정
        output_dir = os.path.join(settings.OUTPUT_DIR, "idm_vton")
        os.makedirs(output_dir, exist_ok=True)
        result_id = str(uuid.uuid4())
        output_path = os.path.join(output_dir, f"{result_id}.jpg")
        
        # 7. Agnostic 이미지 생성 또는 조회
        agnostic_img_path = None
        mask_path = None
        
        if request_data.agnostic_id:
            # 기존 agnostic 이미지 사용
            agnostic_service = get_agnostic_service()
            agnostic_info = agnostic_service.get_agnostic_by_id(request_data.agnostic_id)
            
            if agnostic_info and agnostic_info["status"] == "success":
                agnostic_img_path = agnostic_info["agnostic_path"]
                mask_path = agnostic_info["mask_path"]
                print(f"기존 Agnostic 이미지 사용: {agnostic_img_path}")
            else:
                print(f"지정된 Agnostic ID를 찾을 수 없음: {request_data.agnostic_id}, 새로 생성합니다.")
                request_data.agnostic_id = None
        
        if not request_data.agnostic_id:
            # 새 agnostic 이미지 생성
            agnostic_service = get_agnostic_service()
            agnostic_result = agnostic_service.generate_agnostic(
                image_path=user_img_path,
                parsing_json_path=parsing_result_path,
                keypoints_json_path=os.path.join(settings.OUTPUT_DIR, "mediapipe", f"{request_data.user_image_id}_keypoints.json"),
                category=request_data.category
            )
            
            if agnostic_result["status"] == "success":
                agnostic_img_path = agnostic_result["agnostic_path"]
                mask_path = agnostic_result["mask_path"]
                print(f"새 Agnostic 이미지 생성 완료: {agnostic_img_path}")
            else:
                raise HTTPException(status_code=500, detail=f"Agnostic 이미지 생성 실패: {agnostic_result.get('message', '알 수 없는 오류')}")
        
        # 8. IDM VTON 모델 호출
        print("\n=== IDM VTON 모델 호출 시작 ===")
        print(f"사용자 이미지: {user_img_path}")
        print(f"Agnostic 이미지: {agnostic_img_path}")
        print(f"의류 이미지: {clothing_img_path}")
        print(f"파싱 결과 경로: {parsing_result_path}")
        print(f"카테고리: {request_data.category}")
        
        # DensePose 데이터 확인
        print(f"DensePose 데이터 크기: {len(json.dumps(densepose_data))} 바이트")
        print(f"DensePose 데이터에 포함된 키: {list(densepose_data.keys())}")
        if "uv_coordinates" in densepose_data:
            print(f"UV 좌표 개수: {len(densepose_data.get('uv_coordinates', []))}")
        
        try:
            from app.services.idm_vton_service import get_idm_vton_service
            idm_vton_service = get_idm_vton_service()
            print("IDM VTON 서비스 인스턴스 생성 완료")
            
            # IDM VTON 처리 실행
            print("\n=== process_images 메서드 호출 시작 ===")
            result_path = idm_vton_service.process_images(
                person_image_path=user_img_path,
                clothing_image_path=clothing_img_path,
                parsing_result_path=parsing_result_path,
                densepose_data=densepose_data,
                category=request_data.category,
                agnostic_image_path=agnostic_img_path  # agnostic 이미지 추가
            )
            print(f"IDM VTON 처리 완료, 결과 경로: {result_path}")
        except Exception as e:
            print(f"\n=== IDM VTON 처리 중 오류 발생 ===")
            print(f"오류 메시지: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"IDM VTON 처리 오류: {str(e)}")
        
        # 9. 카테고리별 후처리 적용 (있을 경우)
        if hasattr(idm_vton_service, 'apply_category_specific_processing'):
            result_path = idm_vton_service.apply_category_specific_processing(request_data.category, result_path)
        
        # 10. 결과 반환
        # result_path에서 파일 이름 추출
        result_filename = os.path.basename(result_path)
        file_id = result_id  # 결과 ID 유지
        
        # API 응답용 URL 경로와 실제 파일 경로 일치시키기
        api_url = f"/api/output/idm_vton/{file_id}.jpg"
        
        # API 응답으로 제공할 액세스 가능한 URL
        frontend_url = f"/output/idm_vton/{result_filename}"
        
        return {
            "status": "success",
            "result_id": file_id,
            "image_url": frontend_url,
            "api_url": api_url,
            "file_path": result_path,  # 디버깅용 전체 경로
            "category": request_data.category,
            "message": "IDM VTON 처리 완료",
            "agnostic_id": agnostic_result["agnostic_id"] if 'agnostic_result' in locals() else request_data.agnostic_id
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"IDM VTON 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 후처리 로직은 이제 idm_vton_service.py의 apply_category_specific_processing 메서드로 이동

class GenerateMaskRequest(BaseModel):
    category: str
    keypoints: str
    image_width: int
    image_height: int
    parsing_id: str

@router.post("/generate-mask")
async def generate_mask(
    request_data: GenerateMaskRequest
):
    """
    키포인트 기반 마스크 생성 및 테스트 API
    
    - **category**: 의류 카테고리 (upper, lower, dress, outer)
    - **keypoints**: 미디어파이프 키포인트 정보 (JSON 문자열)
    - **image_width**: 이미지 가로 크기
    - **image_height**: 이미지 세로 크기
    - **parsing_id**: 인체 파싱 결과 ID
    """
    try:
        # 1. 카테고리 검증
        valid_categories = ["upper", "lower", "dress", "outer"]
        if request_data.category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 카테고리입니다. 유효한 값: {', '.join(valid_categories)}"
            )
            
        # 2. 키포인트 정보 파싱
        try:
            keypoints = json.loads(request_data.keypoints)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="키포인트 정보 형식이 잘못되었습니다")
            
        # 3. 인체 파싱 결과 확인
        parsing_result_path = os.path.join(settings.OUTPUT_DIR, "human_parsing", f"{request_data.parsing_id}_mask.json")
        
        if not os.path.exists(parsing_result_path):
            raise HTTPException(status_code=404, detail=f"인체 파싱 결과를 찾을 수 없습니다: {request_data.parsing_id}")
            
        # 4. IDM VTON 서비스 호출하여 마스크 생성
        from app.services.idm_vton_service import get_idm_vton_service
        idm_vton_service = get_idm_vton_service()
        
        # 파싱 결과 로드
        print(f"파싱 결과 로드: {parsing_result_path}")
        with open(parsing_result_path, 'r') as f:
            parsing_data = json.load(f)
            
        # 카테고리에 맞는 타겟 부분 가져오기
        target_parts = idm_vton_service._get_target_parts_for_category(request_data.category)
        print(f"카테고리 {request_data.category}에 대한 타겟 부분: {target_parts}")
        
        # 기본 마스크 생성
        print("기본 마스크 생성 중...")
        initial_mask = idm_vton_service._create_segmentation_mask(parsing_data, target_parts)
        
        # 키포인트 기반 마스크 강화
        print(f"키포인트 기반 마스크 강화 중... (키포인트 수: {len(keypoints)})")
        enhanced_mask = idm_vton_service._enhance_mask_with_keypoints(
            initial_mask, 
            keypoints, 
            target_parts
        )
        
        # 디버깅: 원본 마스크와 개선된 마스크 저장
        output_dir = os.path.join(settings.OUTPUT_DIR, "masks", "debug")
        os.makedirs(output_dir, exist_ok=True)
        
        debug_id = str(uuid.uuid4())[:8]
        initial_mask_path = os.path.join(output_dir, f"{debug_id}_initial_mask.png")
        enhanced_mask_path = os.path.join(output_dir, f"{debug_id}_enhanced_mask.png")
        
        cv2.imwrite(initial_mask_path, initial_mask)
        cv2.imwrite(enhanced_mask_path, enhanced_mask)
        print(f"디버그 이미지 저장: \n - 원본 마스크: {initial_mask_path}\n - 개선 마스크: {enhanced_mask_path}")
        
        # 5. 결과 이미지 저장
        output_dir = os.path.join(settings.OUTPUT_DIR, "masks")
        os.makedirs(output_dir, exist_ok=True)
        
        result_id = str(uuid.uuid4())
        mask_path = os.path.join(output_dir, f"{result_id}_mask.png")
        
        # OpenCV 이미지로 저장
        cv2.imwrite(mask_path, enhanced_mask)
        
        # 6. 결과 반환
        return {
            "status": "success",
            "message": "마스크 생성 완료",
            "category": request_data.category,
            "mask_path": f"/output/masks/{result_id}_mask.png"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"마스크 생성 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))