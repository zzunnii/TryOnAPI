"""
이미지 처리 유틸리티 함수
"""

import os
import uuid
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, List
import io
import base64
import cv2

def resize_image(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> Image.Image:
    """
    이미지 크기 조정

    Args:
        image: 이미지 경로, PIL 이미지 또는 NumPy 배열
        target_size: 목표 크기 (width, height)
        keep_aspect_ratio: 가로세로 비율 유지 여부

    Returns:
        PIL.Image: 크기가 조정된 이미지
    """
    # 이미지 로드
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            img = Image.fromarray(image)
        else:
            img = Image.fromarray((image * 255).astype(np.uint8))
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("지원되지 않는 이미지 형식입니다")

    if keep_aspect_ratio:
        # 비율 유지하면서 크기 조정
        width, height = img.size
        target_width, target_height = target_size

        # 가로세로 비율 계산
        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        if aspect_ratio > target_aspect_ratio:
            # 이미지가 더 넓은 경우
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
        else:
            # 이미지가 더 높은 경우
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        # 이미지 리사이즈
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # 목표 크기의 새 이미지 생성
        result_img = Image.new("RGB", target_size, color=(0, 0, 0))

        # 중앙에 리사이즈된 이미지 붙이기
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result_img.paste(resized_img, (paste_x, paste_y))

        return result_img
    else:
        # 단순 리사이즈
        return img.resize(target_size, Image.LANCZOS)

def save_image(
    image: Union[str, Image.Image, np.ndarray],
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    filename_prefix: str = "img",
    extension: str = "jpg"
) -> str:
    """
    이미지 저장

    Args:
        image: 이미지 경로, PIL 이미지 또는 NumPy 배열
        output_path: 출력 경로 (None이면 자동 생성)
        output_dir: 출력 디렉토리 (output_path가 None일 때 사용)
        filename_prefix: 파일 이름 접두사
        extension: 파일 확장자

    Returns:
        str: 저장된 이미지 경로
    """
    # 이미지 로드
    if isinstance(image, str):
        # 이미 경로가 있는 경우, 해당 경로 반환 또는 복사
        if output_path is None:
            if output_dir is not None:
                # 파일명만 추출
                basename = os.path.basename(image)
                output_path = os.path.join(output_dir, basename)
                # 다른 경로로 복사
                if image != output_path:
                    img = Image.open(image)
                    img.save(output_path)
            else:
                # 그대로 경로 반환
                return image
        else:
            # 다른 경로로 복사
            img = Image.open(image)
            img.save(output_path)
    else:
        # PIL 이미지 또는 NumPy 배열 변환
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                img = Image.fromarray(image)
            else:
                img = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("지원되지 않는 이미지 형식입니다")

        # 출력 경로 생성
        if output_path is None:
            if output_dir is None:
                raise ValueError("output_path 또는 output_dir 중 하나가 필요합니다")

            # 디렉토리 확인 및 생성
            os.makedirs(output_dir, exist_ok=True)

            # 고유 파일명 생성
            unique_id = str(uuid.uuid4())
            output_path = os.path.join(output_dir, f"{filename_prefix}_{unique_id}.{extension}")

        # 이미지 저장
        img.save(output_path)

    return output_path

def image_to_base64(image: Union[str, Image.Image, np.ndarray]) -> str:
    """
    이미지를 Base64 문자열로 변환

    Args:
        image: 이미지 경로, PIL 이미지 또는 NumPy 배열

    Returns:
        str: Base64 인코딩 문자열
    """
    # 이미지 로드
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("지원되지 않는 이미지 형식입니다")

    # 바이트 스트림으로 변환
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    # Base64 인코딩
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_str

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Base64 문자열을 이미지로 변환

    Args:
        base64_str: Base64 인코딩 문자열

    Returns:
        PIL.Image: 디코딩된 이미지
    """
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def create_mask_from_segmentation(
    segmentation_data: dict,
    target_parts: list,
    image_size: Tuple[int, int] = (768, 1024)
) -> Image.Image:
    """
    세그멘테이션 데이터에서 마스크 생성

    Args:
        segmentation_data: 세그멘테이션 결과 데이터
        target_parts: 마스크에 포함할 타겟 부분 목록
        image_size: 마스크 이미지 크기 (기본값: 768x1024)

    Returns:
        PIL.Image: 생성된 마스크 이미지
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    # 세그멘테이션 데이터에서 타겟 부분 추출
    for part_id in target_parts:
        if str(part_id) in segmentation_data:
            for point in segmentation_data[str(part_id)]:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 255

    # 마스크 정제 (선택적)
    # 필요에 따라 dilate, erode, blur 등 추가 처리 가능
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # PIL 이미지로 변환
    return Image.fromarray(mask)

def resize_to_original(
        image,
        original_size: Tuple[int, int],
        resample=Image.NEAREST
) -> Image.Image:
    """
    이미지를 원본 크기로 리사이즈

    Parameters:
    -----------
    image : PIL.Image
        리사이즈할 이미지
    original_size : tuple
        원본 이미지 크기 (width, height)
    resample : int
        리샘플링 방법

    Returns:
    --------
    PIL.Image
        리사이즈된 이미지
    """
    return image.resize(original_size, resample)


def combine_masks(
        masks: List[Image.Image],
        mode: str = "max"
) -> Image.Image:
    """
    여러 마스크를 결합

    Parameters:
    -----------
    masks : list
        결합할 마스크 이미지 목록
    mode : str
        결합 모드 ('max', 'min', 'avg')

    Returns:
    --------
    PIL.Image
        결합된 마스크 이미지
    """
    if not masks:
        return None

    # 모든 마스크를 numpy 배열로 변환
    arrays = [np.array(mask) for mask in masks]

    # 결합 모드에 따라 처리
    if mode == "max":
        result = np.max(arrays, axis=0)
    elif mode == "min":
        result = np.min(arrays, axis=0)
    elif mode == "avg":
        result = np.mean(arrays, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"지원하지 않는 결합 모드: {mode}")

    return Image.fromarray(result)