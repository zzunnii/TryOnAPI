import os
import numpy as np
import cv2
import uuid
from PIL import Image
from typing import Tuple, Any

def generate_unique_id() -> str:
    """
    고유 ID 생성
    """
    return str(uuid.uuid4())

def create_directory(directory_path: str) -> None:
    """
    디렉토리 생성 (존재하지 않는 경우)
    """
    os.makedirs(directory_path, exist_ok=True)

def resize_image(image: Any, target_size: Tuple[int, int]) -> np.ndarray:
    """
    이미지 크기 조정
    """
    if isinstance(image, str):
        # 이미지 경로인 경우
        image = cv2.imread(image)
    
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

def pad_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    이미지를 패딩하여 목표 크기로 만듦
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 패딩 계산
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 패딩 적용
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

def crop_center(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    이미지 중앙 크롭
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 크롭 영역 계산
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)
    
    # 크롭 적용
    return image[start_h:start_h + target_h, start_w:start_w + target_w]

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    이미지에 마스크 적용
    """
    # 마스크가 단일 채널인 경우 3채널로 확장
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
    
    # 마스크 적용
    return cv2.bitwise_and(image, mask)

def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    두 이미지를 블렌딩
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)