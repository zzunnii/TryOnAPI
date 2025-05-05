"""
TryOnAPI 테스트 함수
플러터 앱에서의 API 호출을 시뮬레이션하는 데모 코드
"""

import os
import requests
import json
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import cv2
import uuid

# 서버 URL 설정
API_URL = "http://localhost:8010"  # 로컬 테스트용
# API_URL = "http://your-production-server.com"  # 프로덕션 서버

# 설정 가져오기 (app/core/config.py에서 가져오기)
from app.core.config import get_settings
settings = get_settings()

def display_image(img, title="Image"):
    """이미지 표시 helper 함수"""
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.close()

def upload_image(image_path, endpoint):
    """이미지 업로드 함수"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as file:
        files = {'file': (os.path.basename(image_path), file, 'image/jpeg')}
        response = requests.post(f"{API_URL}/api{endpoint}", files=files)
    
    if response.status_code != 200:
        print(f"업로드 실패: {response.status_code}")
        print(response.text)
        return None
    
    return response.json()

def process_human_parsing(image_path, save_visualization=True):
    """인체 파싱 처리 함수"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as file:
        files = {'image': (os.path.basename(image_path), file, 'image/jpeg')}
        data = {
            'save_visualization': str(save_visualization).lower(),
            'use_firebase': 'false'
        }
        response = requests.post(f"{API_URL}/api/human-parsing/process", files=files, data=data)

    if response.status_code != 200:
        print(f"인체 파싱 처리 실패: {response.status_code}")
        print(response.text)
        return None
    
    return response.json()

def generate_mock_densepose_response(image_path):
    """목업 DensePose 응답 생성"""
    try:
        # 이미지 크기 가져오기
        img = Image.open(image_path)
        width, height = img.size
        
        # 가짜 데이터 생성
        num_points = 500  # UV 좌표 수
        
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
        
        return {
            "status": "success",
            "densepose_data": {
                "segments": segments,
                "uv_coordinates": uv_coordinates,
                "image_width": width,
                "image_height": height
            }
        }
    
    except Exception as e:
        print(f"목업 DensePose 응답 생성 오류: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_densepose(image_path):
    """DensePose API 호출 (Detectron2 API 서버)"""
    print("\n=== Detectron2 DensePose API 호출 중... ===")
    
    # 이미지 파일 확인
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # Detectron2 DensePose API URL
    DENSEPOSE_API_URL = "http://localhost:8001/process/file"
    
    try:
        # 이미지 파일 업로드
        with open(image_path, 'rb') as img_file:
            files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}
            params = {
                'include_body_parts': 'true',
                'include_uv_coordinates': 'true',
                'create_visualization': 'true'
            }
            
            # API 요청 실행
            try:
                response = requests.post(DENSEPOSE_API_URL, files=files, params=params)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"DensePose API 호출 오류: {response.status_code}, {response.text}")
            except requests.RequestException as e:
                print(f"DensePose API 서버 연결 오류: {str(e)}")
                print("Detectron2 API 서버가 실행 중인지 확인해주세요.")
        
        # API 호출 실패 시 목업 응답 사용
        print("실제 API 호출 실패 - 목업 응답 사용 (테스트용)")
        time.sleep(1)  # API 호출 시뮬레이션
        
        # 목업 응답 생성
        mock_response = generate_mock_densepose_response(image_path)
        return mock_response
        
    except Exception as e:
        print(f"DensePose API 호출 오류: {str(e)}")
        return {"status": "error", "message": str(e)}

def generate_agnostic_image(person_image_path, parsing_id, mediapipe_keypoints_path, category, result_dir):
    """
    Agnostic 이미지 생성 API 호출
    app/utils/agnostic_utils.py의 함수 사용
    
    Args:
        person_image_path: 사용자 이미지 경로
        parsing_id: 인체 파싱 결과 ID
        mediapipe_keypoints_path: 미디어파이프 키포인트 JSON 파일 경로
        category: 의류 카테고리
        result_dir: 결과 저장 디렉토리
        
    Returns:
        dict: Agnostic 이미지 정보 (agnostic_id, agnostic_path 등)
    """
    print("\n=== Agnostic 이미지 생성 시작 ===")
    
    try:
        # 1. 파싱 결과 및 키포인트 파일 확인
        parsing_result_path = os.path.join(settings.OUTPUT_DIR, "human_parsing", f"{parsing_id}_mask.json")
        if not os.path.exists(parsing_result_path):
            raise FileNotFoundError(f"파싱 결과를 찾을 수 없습니다: {parsing_result_path}")
        
        # 2. agnostic_utils.py의 함수 사용
        from app.utils.agnostic_utils import create_agnostic_image
        
        # 결과 저장 경로
        agnostic_output_dir = os.path.join(settings.OUTPUT_DIR, "agnostic")
        os.makedirs(agnostic_output_dir, exist_ok=True)
        
        agnostic_id = str(uuid.uuid4())
        agnostic_path = os.path.join(agnostic_output_dir, f"{agnostic_id}_agnostic.jpg")
        
        # create_agnostic_image 함수 사용하여 agnostic 이미지 생성
        agnostic_image, idm_mask = create_agnostic_image(
            image_path=person_image_path,
            parsing_json_path=parsing_result_path,
            keypoints_json_path=mediapipe_keypoints_path,
            category=category,
            output_path=agnostic_path
        )
        
        # 결과 저장
        mask_path = agnostic_path.replace('.jpg', '_mask.png')
        
        # 사용자 폴더에 복사본 저장 (확인용)
        if result_dir:
            user_agnostic_path = os.path.join(result_dir, f"agnostic_{category}.jpg")
            user_mask_path = os.path.join(result_dir, f"agnostic_mask_{category}.png")
            
            agnostic_image.save(user_agnostic_path)
            cv2.imwrite(user_mask_path, idm_mask * 255)
            print(f"사용자 폴더에 복사본 저장: {user_agnostic_path}")
        
        print(f"Agnostic 이미지 생성 완료: {agnostic_path}")
        
        # 결과 반환
        return {
            "status": "success",
            "agnostic_id": agnostic_id,
            "agnostic_path": agnostic_path,
            "mask_path": mask_path,
            "category": category
        }
        
    except Exception as e:
        print(f"Agnostic 이미지 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

def try_on_with_idm_vton(
    person_image_path, 
    clothing_image_path, 
    category, 
    output_path=None, 
    save_mask_only=False, 
    skip_parsing=False,
    generate_agnostic=True,    # Agnostic 이미지 생성 여부
    reuse_agnostic_id=None     # 재사용할 Agnostic 이미지 ID
):
    """
    IDM VTON을 사용한 가상 착용 처리
    
    Args:
        person_image_path: 사용자 이미지 경로
        clothing_image_path: 의류 이미지 경로
        category: 의류 카테고리 (upper, lower, dress, outer)
        output_path: 결과 이미지 저장 경로
        save_mask_only: True인 경우 마스크만 저장하고 가상 착용은 건너뜀
        skip_parsing: True인 경우 인체 파싱을 건너뛰고 테스트만 진행
        generate_agnostic: True인 경우 Agnostic 이미지 생성 (기본값: True)
        reuse_agnostic_id: 재사용할 Agnostic 이미지 ID (있을 경우)
    """
    if not os.path.exists(person_image_path) or not os.path.exists(clothing_image_path):
        raise FileNotFoundError("이미지 파일을 찾을 수 없습니다")
    
    # 결과 저장 디렉토리 생성
    result_dir = os.path.dirname(output_path) if output_path else './result'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n=== IDM VTON 가상 착용 시작 ===")
    print(f"사용자 이미지: {person_image_path}")
    print(f"의류 이미지: {clothing_image_path}")
    print(f"카테고리: {category}")
    print(f"결과 저장 경로: {result_dir}")
    
    if save_mask_only:
        print("모드: 마스크 생성 및 디버그")
    if skip_parsing:
        print("모드: 인체 파싱 건너뛰기 (테스트용)")
    if generate_agnostic:
        print("모드: Agnostic 이미지 생성 활성화")
    if reuse_agnostic_id:
        print(f"모드: Agnostic 이미지 재사용 (ID: {reuse_agnostic_id})")
    
    # 원본 이미지 복사 (중간 처리 과정 확인용)
    orig_person_path = os.path.join(result_dir, "original_person.jpg")
    orig_cloth_path = os.path.join(result_dir, "original_clothing.jpg")
    
    # 원본 이미지 복사
    try:
        from shutil import copyfile
        copyfile(person_image_path, orig_person_path)
        copyfile(clothing_image_path, orig_cloth_path)
        print(f"원본 이미지 저장: {orig_person_path}, {orig_cloth_path}")
    except Exception as e:
        print(f"원본 이미지 복사 실패: {str(e)}")
    
    # 디버그: 필요한 디렉토리 확인
    print("\n=== 디렉토리 구조 확인 ===")
    for dir_path in [
        os.path.join(settings.TEMP_DIR, "user_images"),
        os.path.join(settings.TEMP_DIR, "target_clothing"),
        os.path.join(settings.OUTPUT_DIR, "human_parsing"),
        os.path.join(settings.OUTPUT_DIR, "idm_vton"),
        os.path.join(settings.OUTPUT_DIR, "mediapipe"),
        os.path.join(settings.OUTPUT_DIR, "agnostic"),  # 추가: agnostic 이미지 디렉토리
    ]:
        exists = os.path.exists(dir_path)
        print(f"디렉토리 {dir_path}: {'존재함' if exists else '존재하지 않음'}")
        if not exists:
            os.makedirs(dir_path, exist_ok=True)
            print(f"  -> 디렉토리 생성됨: {dir_path}")
    
    # 모델 디렉토리 확인
    model_dir = os.path.join(settings.BASE_DIR, "models", "idm_vton")
    print(f"\n모델 디렉토리 확인: {model_dir}")
    if os.path.exists(model_dir):
        contents = os.listdir(model_dir)
        print(f"  -> 디렉토리 내용: {contents}")
    else:
        print("  -> 모델 디렉토리가 존재하지 않습니다. setup_model.py를 실행하세요.")
        os.makedirs(model_dir, exist_ok=True)
        print(f"  -> 디렉토리 생성됨: {model_dir}")
    
    # 파싱 결과 ID (인체 파싱을 건너뛰는 경우 임의로 생성)
    parsing_id = None
    parsing_result_path = None
    
    if not skip_parsing:
        # 1. 인체 파싱 처리
        print("\n1. 인체 파싱 처리 중...")
        parsing_result = process_human_parsing(person_image_path)
        if not parsing_result:
            return None
        
        parsing_id = parsing_result['file_id']
        print(f"인체 파싱 완료: {parsing_id}")
        
        # 파싱 결과 이미지 저장
        try:
            parsing_vis_url = parsing_result.get('visualization_path')
            if parsing_vis_url:
                parsing_img_response = requests.get(f"{API_URL}{parsing_vis_url}")
                if parsing_img_response.status_code == 200:
                    parsing_vis_path = os.path.join(result_dir, "parsing_visualization.png")
                    with open(parsing_vis_path, 'wb') as f:
                        f.write(parsing_img_response.content)
                    print(f"파싱 시각화 이미지 저장: {parsing_vis_path}")
        except Exception as e:
            print(f"파싱 결과 이미지 저장 실패: {str(e)}")
    else:
        # 인체 파싱을 건너뛰고 임의의 마스크 생성 (테스트용)
        print("\n1. 인체 파싱 건너뛰기 (테스트 모드)")
        
        # 임의의 파싱 ID 생성
        parsing_id = "test_" + str(uuid.uuid4())
        
        # 테스트용 마스크 JSON 생성
        img = cv2.imread(person_image_path)
        height, width = img.shape[:2]
        
        # 중앙 영역을 타겟으로 설정 (예: 상의)
        mask_json = {
            "image_size": {"width": width, "height": height},
            "class_masks": {
                "inner_torso": [],
                "inner_rsleeve": [],
                "inner_lsleeve": [],
                "neck": []
            }
        }
        
        # 간단한 마스크 만들기
        if category == "upper" or category == "outer":
            # 상체 영역 (상의)
            for y in range(height // 3, height // 2):
                for x in range(width // 3, width * 2 // 3):
                    mask_json["class_masks"]["inner_torso"].append([x, y])
            
            # 소매 영역
            for y in range(height // 3, height // 2):
                # 왼쪽 소매
                for x in range(width // 6, width // 3):
                    mask_json["class_masks"]["inner_lsleeve"].append([x, y])
                # 오른쪽 소매
                for x in range(width * 2 // 3, width * 5 // 6):
                    mask_json["class_masks"]["inner_rsleeve"].append([x, y])
            
            # 목 영역
            for y in range(height // 4, height // 3):
                for x in range(width * 2 // 5, width * 3 // 5):
                    mask_json["class_masks"]["neck"].append([x, y])
                    
        elif category == "lower":
            # 하체 영역 (하의)
            mask_json["class_masks"] = {
                "pants_hip": [],
                "pants_rsleeve": [],
                "pants_lsleeve": []
            }
            
            # 허리 영역
            for y in range(height // 2, height * 3 // 5):
                for x in range(width // 3, width * 2 // 3):
                    mask_json["class_masks"]["pants_hip"].append([x, y])
            
            # 바지 다리 영역
            for y in range(height * 3 // 5, height * 4 // 5):
                # 왼쪽 다리
                for x in range(width * 3 // 8, width // 2):
                    mask_json["class_masks"]["pants_lsleeve"].append([x, y])
                # 오른쪽 다리
                for x in range(width // 2, width * 5 // 8):
                    mask_json["class_masks"]["pants_rsleeve"].append([x, y])
        
        elif category == "dress" or category == "long_dress":
            # 드레스 (상+하체 결합)
            mask_json["class_masks"] = {
                "inner_torso": [],
                "inner_rsleeve": [],
                "inner_lsleeve": [],
                "neck": [],
                "pants_hip": [],
                "skirt": []
            }
            
            # 상체 영역
            for y in range(height // 3, height // 2):
                for x in range(width // 3, width * 2 // 3):
                    mask_json["class_masks"]["inner_torso"].append([x, y])
            
            # 소매 영역
            for y in range(height // 3, height // 2):
                # 왼쪽 소매
                for x in range(width // 6, width // 3):
                    mask_json["class_masks"]["inner_lsleeve"].append([x, y])
                # 오른쪽 소매
                for x in range(width * 2 // 3, width * 5 // 6):
                    mask_json["class_masks"]["inner_rsleeve"].append([x, y])
            
            # 목 영역
            for y in range(height // 4, height // 3):
                for x in range(width * 2 // 5, width * 3 // 5):
                    mask_json["class_masks"]["neck"].append([x, y])
            
            # 하체 영역
            for y in range(height // 2, height * 4 // 5):
                for x in range(width // 3, width * 2 // 3):
                    if y < height * 3 // 5:
                        mask_json["class_masks"]["pants_hip"].append([x, y])
                    else:
                        mask_json["class_masks"]["skirt"].append([x, y])
        
        # 마스크 JSON 저장
        output_dir = os.path.join(settings.OUTPUT_DIR, "human_parsing")
        os.makedirs(output_dir, exist_ok=True)
        parsing_result_path = os.path.join(output_dir, f"{parsing_id}_mask.json")
        
        with open(parsing_result_path, 'w') as f:
            json.dump(mask_json, f, indent=2)
        
        print(f"테스트용 마스크 JSON 생성 완료: {parsing_result_path}")
        
        # 테스트용 파싱 결과 생성
        parsing_result = {
            "status": "success",
            "file_id": parsing_id,
            "message": "테스트용 파싱 결과"
        }
    
    # 2. 미디어파이프 키포인트 처리 (옵션)
    print("\n2. 미디어파이프 키포인트 추출 중...")
    mediapipe_keypoints_path = None
    try:
        mediapipe_response = requests.post(
            f"{API_URL}/api/mediapipe/keypoints", 
            files={'file': open(person_image_path, 'rb')}
        )
        
        if mediapipe_response.status_code == 200:
            mediapipe_result = mediapipe_response.json()
            mediapipe_landmarks = mediapipe_result.get('pose_landmarks', [])
            print(f"미디어파이프 키포인트 추출 성공: {len(mediapipe_landmarks)} 개")
            
            # 시각화 이미지 저장
            vis_url = mediapipe_result.get('visualization_path')
            if vis_url:
                vis_response = requests.get(f"{API_URL}{vis_url}")
                if vis_response.status_code == 200:
                    keypoints_vis_path = os.path.join(result_dir, "mediapipe_keypoints.png")
                    with open(keypoints_vis_path, 'wb') as f:
                        f.write(vis_response.content)
                    print(f"키포인트 시각화 이미지 저장: {keypoints_vis_path}")
            
            # 키포인트 JSON 파일 경로 저장
            mediapipe_keypoints_path = os.path.join(settings.OUTPUT_DIR, "mediapipe", f"{mediapipe_result['file_id']}_keypoints.json")
            
            # 마스크 테스트 모드인 경우 키포인트 기반 마스크 생성
            if save_mask_only and mediapipe_landmarks:
                # 마스크 생성 전용 API 엔드포인트 호출 
                print("\n2.1 키포인트 기반 마스크 생성 테스트 중...")
                try:
                    mask_test_data = {
                        "category": category,
                        "keypoints": json.dumps(mediapipe_landmarks),
                        "image_width": mediapipe_result.get('image_width', 512),
                        "image_height": mediapipe_result.get('image_height', 768),
                        "parsing_id": parsing_result['file_id']
                    }
                    
                    # 마스크 생성 API 호출 (디버그용)
                    mask_response = requests.post(f"{API_URL}/api/tryon/generate-mask", json=mask_test_data)
                    
                    if mask_response.status_code == 200:
                        mask_result = mask_response.json()
                        
                        # 마스크 이미지 저장
                        if 'mask_path' in mask_result:
                            mask_img_response = requests.get(f"{API_URL}{mask_result['mask_path']}")
                            if mask_img_response.status_code == 200:
                                mask_path = os.path.join(result_dir, f"mask_{category}.png")
                                with open(mask_path, 'wb') as f:
                                    f.write(mask_img_response.content)
                                print(f"키포인트 기반 마스크 저장: {mask_path}")
                                
                                # 마스크와 원본 이미지 합성하여 시각화
                                try:
                                    person = cv2.imread(person_image_path)
                                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                    
                                    # 이미지 크기 맞추기
                                    mask = cv2.resize(mask, (person.shape[1], person.shape[0]))
                                    
                                    # 마스크를 컬러 형식으로 변환
                                    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                                    mask_color[:, :, 0] = 0  # 파란색 채널 제거
                                    mask_color[:, :, 2] = 0  # 빨간색 채널 제거
                                    
                                    # 마스크 영역만 강조
                                    alpha = 0.5
                                    overlay = cv2.addWeighted(person, 1.0, mask_color, alpha, 0)
                                    
                                    # 합성 이미지 저장
                                    overlay_path = os.path.join(result_dir, f"overlay_{category}.jpg")
                                    cv2.imwrite(overlay_path, overlay)
                                    print(f"마스크 오버레이 이미지 저장: {overlay_path}")
                                except Exception as e:
                                    print(f"마스크 오버레이 생성 실패: {str(e)}")
                    else:
                        print(f"마스크 생성 API 호출 실패: {mask_response.status_code}")
                except Exception as e:
                    print(f"마스크 테스트 중 오류 발생: {str(e)}")
        else:
            print(f"미디어파이프 키포인트 추출 실패: {mediapipe_response.status_code}")
    except Exception as e:
        print(f"미디어파이프 처리 오류 (무시됨): {str(e)}")
    
    # 마스크 생성 모드인 경우 여기서 종료
    if save_mask_only:
        print("\n=== 마스크 생성 모드: 가상 착용 처리 건너뜀 ===")
        return {
            "status": "success",
            "message": "마스크 생성 모드 완료",
            "category": category
        }
    
    # 3. DensePose 정보 가져오기 (외부 API)
    print("\n3. DensePose 처리 중...")
    densepose_result = get_densepose(person_image_path)
    if not densepose_result or densepose_result['status'] != 'success':
        print("DensePose 처리 실패")
        return None
    
    print("DensePose 처리 완료")
    
    # 4. 사용자 이미지 업로드
    print("\n4. 사용자 이미지 업로드 중...")
    user_upload_result = upload_image(person_image_path, "/tryon/upload/user-image")
    if not user_upload_result:
        return None
    
    user_file_id = user_upload_result['file_id']
    print(f"사용자 이미지 업로드 완료: {user_file_id}")
    
    # 5. 의류 이미지 업로드
    print("\n5. 의류 이미지 업로드 중...")
    clothing_upload_result = upload_image(clothing_image_path, "/tryon/upload/clothing")
    if not clothing_upload_result:
        return None
    
    clothing_file_id = clothing_upload_result['file_id']
    print(f"의류 이미지 업로드 완료: {clothing_file_id}")
    
    # 미디어파이프 키포인트가 있으면 DensePose 데이터에 추가
    if 'mediapipe_result' in locals() and mediapipe_result.get('pose_landmarks'):
        densepose_result['densepose_data']['keypoints'] = mediapipe_result.get('pose_landmarks', [])
        print("DensePose 데이터에 미디어파이프 키포인트 추가")
    
    # 6. Agnostic 이미지 생성 (필요한 경우)
    agnostic_id = None
    if generate_agnostic or reuse_agnostic_id:
        if reuse_agnostic_id:
            # 기존 agnostic 이미지 ID 사용
            agnostic_id = reuse_agnostic_id
            print(f"기존 Agnostic 이미지 ID 사용: {agnostic_id}")
        else:
            # 새 agnostic 이미지 생성
            print("\n6. Agnostic 이미지 생성 중...")
            agnostic_result = generate_agnostic_image(
                person_image_path=person_image_path,
                parsing_id=parsing_result['file_id'],
                mediapipe_keypoints_path=mediapipe_keypoints_path,
                category=category,
                result_dir=result_dir
            )
            
            if agnostic_result and agnostic_result["status"] == "success":
                agnostic_id = agnostic_result["agnostic_id"]
                print(f"새 Agnostic 이미지 생성 완료: {agnostic_id}")
            else:
                print("Agnostic 이미지 생성 실패, 가상 착용은 원본 이미지로 진행합니다.")
    
    # 7. IDM VTON API 호출
    print("\n7. IDM VTON API 호출 중...")
    try_on_data = {
        "user_image_id": user_file_id,
        "clothing_id": clothing_file_id,
        "parsing_id": parsing_result['file_id'],
        "category": category,
        "densepose_data": json.dumps(densepose_result['densepose_data'])
    }
    
    # Agnostic 이미지 ID가 있으면 요청에 추가
    if agnostic_id:
        try_on_data["agnostic_id"] = agnostic_id
        print(f"Agnostic 이미지 ID 