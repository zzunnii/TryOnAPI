"""
IDM-VTON API 테스트 스크립트

이 스크립트는 IDM-VTON API의 기능을 테스트하기 위한 도구입니다.
명령줄에서 실행하거나 함수를 직접 호출하여 API를 테스트할 수 있습니다.
"""

import os
import requests
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from typing import Optional, Dict, Any, List

# 기본 API URL 설정
API_URL = ""

def display_image(img_path, title="Image"):
    """이미지 표시 함수"""
    img = Image.open(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))
    plt.title(title)
    plt.axis('off')
    plt.show()

def upload_user_image(api_url: str, image_path: str) -> Dict[str, Any]:
    """
    사용자 이미지 업로드 함수
    
    Args:
        api_url: API 서버 URL
        image_path: 이미지 파일 경로
        
    Returns:
        Dict: API 응답 데이터
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{api_url}/api/v1/user/upload-image", files=files)
    
    if response.status_code != 200:
        print(f"업로드 실패: {response.status_code}")
        print(response.text)
        return None
    
    print(f"사용자 이미지 업로드 성공: {os.path.basename(image_path)}")
    return response.json()

def upload_clothing(api_url: str, image_path: str, category: str = "upper") -> Dict[str, Any]:
    """
    의류 이미지 업로드 함수
    
    Args:
        api_url: API 서버 URL
        image_path: 이미지 파일 경로
        category: 의류 카테고리 (upper, lower, dress, outer)
        
    Returns:
        Dict: API 응답 데이터
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'category': category}
        response = requests.post(f"{api_url}/api/v1/clothing/upload", files=files, data=data)
    
    if response.status_code != 200:
        print(f"업로드 실패: {response.status_code}")
        print(response.text)
        return None
    
    print(f"의류 이미지 업로드 성공: {os.path.basename(image_path)} (카테고리: {category})")
    return response.json()

def process_human_parsing(api_url: str, image_path: str, save_visualization: bool = True) -> Dict[str, Any]:
    """
    인체 파싱 처리 함수
    
    Args:
        api_url: API 서버 URL
        image_path: 이미지 파일 경로
        save_visualization: 시각화 이미지 저장 여부
        
    Returns:
        Dict: API 응답 데이터
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'save_visualization': str(save_visualization).lower()}
        response = requests.post(f"{api_url}/api/v1/preprocessing/human-parsing", files=files, data=data)
    
    if response.status_code != 200:
        print(f"인체 파싱 처리 실패: {response.status_code}")
        print(response.text)
        return None
    
    print(f"인체 파싱 처리 성공: {os.path.basename(image_path)}")
    return response.json()

def process_mask_generation(
    api_url: str, 
    image_path: str, 
    category: str, 
    parsing_id: Optional[str] = None,
    pose_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    마스크 생성 함수
    
    Args:
        api_url: API 서버 URL
        image_path: 이미지 파일 경로
        category: 의류 카테고리 (upper, lower, dress, outer)
        parsing_id: 인체 파싱 결과 ID (없으면 자동 파싱)
        pose_id: 포즈 추정 결과 ID (없으면 자동 추정)
        
    Returns:
        Dict: API 응답 데이터
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'category': category
        }
        
        if parsing_id:
            data['parsing_id'] = parsing_id
        
        if pose_id:
            data['pose_id'] = pose_id
            
        response = requests.post(f"{api_url}/api/v1/preprocessing/mask-generation", files=files, data=data)
    
    if response.status_code != 200:
        print(f"마스크 생성 처리 실패: {response.status_code}")
        print(response.text)
        return None
    
    print(f"마스크 생성 처리 성공: {os.path.basename(image_path)} (카테고리: {category})")
    return response.json()

def process_tryon(
    api_url: str,
    user_image_id: str,
    clothing_id: str,
    category: str,
    garment_description: Optional[str] = None,
    parsing_id: Optional[str] = None,
    pose_id: Optional[str] = None,
    mask_id: Optional[str] = None,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    guidance_scale: float = 3.0
) -> Dict[str, Any]:
    """
    가상 피팅 처리 함수

    Args:
        api_url: API 서버 URL
        user_image_id: 사용자 이미지 ID
        clothing_id: 의류 이미지 ID
        category: 의류 카테고리 (upper, lower, dress, outer)
        garment_description: 의류 설명 (없으면 카테고리 기반 기본값 사용)
        parsing_id: 인체 파싱 결과 ID (없으면 자동 파싱)
        pose_id: 포즈 결과 ID (없으면 자동 생성)
        mask_id: 마스크 이미지 ID (없으면 자동 생성)
        num_inference_steps: 추론 단계 수 (기본값: 30)
        seed: 랜덤 시드 (없으면 랜덤 사용)
        guidance_scale: 가이던스 스케일 (기본값: 2.0)

    Returns:
        Dict: API 응답 데이터
    """
    data = {
        "user_image_id": user_image_id,
        "clothing_id": clothing_id,
        "category": category
    }

    if garment_description:
        data["garment_description"] = garment_description

    if parsing_id:
        data["parsing_id"] = parsing_id

    if pose_id:
        data["pose_id"] = pose_id

    if mask_id:
        data["mask_id"] = mask_id

    if num_inference_steps != 30:
        data["num_inference_steps"] = num_inference_steps

    if seed is not None:
        data["seed"] = seed

    if guidance_scale != 2.0:
        data["guidance_scale"] = guidance_scale

    start_time = time.time()
    response = requests.post(f"{api_url}/api/v1/tryon/process", json=data)
    processing_time = time.time() - start_time

    if response.status_code != 200:
        print(f"가상 피팅 처리 실패: {response.status_code}")
        print(response.text)
        return None

    print(f"가상 피팅 처리 성공 (처리 시간: {processing_time:.2f}초)")
    return response.json()

def download_result_image(api_url: str, result_id: str, output_path: str) -> bool:
    """
    결과 이미지 다운로드 함수

    Args:
        api_url: API 서버 URL
        result_id: 결과 ID
        output_path: 출력 파일 경로

    Returns:
        bool: 다운로드 성공 여부
    """
    response = requests.get(f"{api_url}/api/v1/tryon/result/{result_id}")

    if response.status_code != 200:
        print(f"결과 이미지 다운로드 실패: {response.status_code}")
        print(response.text)
        return False

    # 결과 이미지 저장
    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"결과 이미지 다운로드 성공: {output_path}")
    return True

def run_end_to_end_test(
    api_url: str,
    person_image_path: str,
    clothing_image_path: str,
    category: str = "upper",
    garment_description: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    엔드투엔드 테스트 실행 함수

    Args:
        api_url: API 서버 URL
        person_image_path: 사용자 이미지 파일 경로
        clothing_image_path: 의류 이미지 파일 경로
        category: 의류 카테고리 (upper, lower, dress, outer)
        garment_description: 의류 설명 (없으면 카테고리 기반 기본값 사용)
        output_path: 결과 이미지 저장 경로 (없으면 자동 생성)

    Returns:
        Dict: 테스트 결과 데이터
    """
    try:
        print(f"\n{'=' * 50}")
        print(f"IDM-VTON API 엔드투엔드 테스트 시작")
        print(f"{'=' * 50}")

        # 시작 시간 기록
        start_time = time.time()

        # 1. 사용자 이미지 업로드
        print("\n[1/4] 사용자 이미지 업로드 중...")
        user_result = upload_user_image(api_url, person_image_path)
        if not user_result:
            return {"status": "error", "message": "사용자 이미지 업로드 실패"}

        user_image_id = user_result["file_id"]

        # 2. 의류 이미지 업로드
        print("\n[2/4] 의류 이미지 업로드 중...")
        clothing_result = upload_clothing(api_url, clothing_image_path, category)
        if not clothing_result:
            return {"status": "error", "message": "의류 이미지 업로드 실패"}

        clothing_id = clothing_result["clothing_id"]

        # 3. 인체 파싱 처리
        print("\n[3/4] 인체 파싱 처리 중...")
        parsing_result = process_human_parsing(api_url, person_image_path)
        if not parsing_result:
            return {"status": "error", "message": "인체 파싱 처리 실패"}

        parsing_id = parsing_result["file_id"]

        # 4. 가상 피팅 처리
        print("\n[4/4] 가상 피팅 처리 중...")
        tryon_result = process_tryon(
            api_url=api_url,
            user_image_id=user_image_id,
            clothing_id=clothing_id,
            category=category,
            garment_description=garment_description,
            parsing_id=parsing_id
        )

        if not tryon_result:
            return {"status": "error", "message": "가상 피팅 처리 실패"}

        result_id = tryon_result["result_id"]

        # 5. 결과 이미지 다운로드
        if not output_path:
            output_dir = "./result"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"tryon_result_{category}_{result_id}.jpg")

        download_success = download_result_image(api_url, result_id, output_path)

        # 처리 시간 계산
        total_time = time.time() - start_time

        # 결과 데이터 구성
        result_data = {
            "status": "success" if download_success else "partial_success",
            "user_image_id": user_image_id,
            "clothing_id": clothing_id,
            "parsing_id": parsing_id,
            "result_id": result_id,
            "category": category,
            "output_path": output_path if download_success else None,
            "processing_time": f"{total_time:.2f}초"
        }

        print(f"\n{'=' * 50}")
        print(f"테스트 완료 (총 처리 시간: {total_time:.2f}초)")
        print(f"결과 이미지: {output_path}")
        print(f"{'=' * 50}")

        return result_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"테스트 중 오류 발생: {str(e)}"}

def main():
    """명령줄 인터페이스 함수"""
    parser = argparse.ArgumentParser(description="IDM-VTON API 테스트 스크립트")
    parser.add_argument("--api-url", type=str, default=API_URL, help="API 서버 URL")
    parser.add_argument("--person", type=str, default=r"./test.jpg", help="사용자 이미지 파일 경로")
    parser.add_argument("--clothing", type=str, default=r"./cloth_test1.jpg", help="의류 이미지 파일 경로")
    parser.add_argument("--category", type=str, default="upper", choices=["upper", "lower", "dress", "outer"], help="의류 카테고리")
    parser.add_argument("--description", type=str, help="의류 설명 (옵션)")
    parser.add_argument("--output", type=str, help="결과 이미지 저장 경로 (옵션)")
    parser.add_argument("--show", action="store_true", default=True, help="결과 이미지 표시 여부")
    
    args = parser.parse_args()
    
    # 파일 경로 확인
    if not os.path.exists(args.person):
        print(f"사용자 이미지 파일을 찾을 수 없습니다: {args.person}")
        return
    
    if not os.path.exists(args.clothing):
        print(f"의류 이미지 파일을 찾을 수 없습니다: {args.clothing}")
        return
    
    # 엔드투엔드 테스트 실행
    result = run_end_to_end_test(
        api_url=args.api_url,
        person_image_path=args.person,
        clothing_image_path=args.clothing,
        category=args.category,
        garment_description=args.description,
        output_path=args.output
    )
    
    # 결과 출력
    print("\n테스트 결과:")
    print(json.dumps(result, indent=2))
    
    # 결과 이미지 표시
    if args.show and result["status"] == "success" and result["output_path"]:
        display_image(result["output_path"], "Virtual Try-On Result")

if __name__ == "__main__":
    main()