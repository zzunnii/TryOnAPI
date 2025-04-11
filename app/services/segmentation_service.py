import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Human Parsing 모델 import
from ..models.parsingModels.parsing_model import ParsingModel
from app.core.config import get_settings
from app.services.human_parsing.processor import get_human_parsing_processor

settings = get_settings()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 세그멘테이션 모델 캐시 (필요할 때만 로드하도록)
_segmentation_model = None


def get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is None:
        from app.services.human_parsing.processor import get_human_parsing_processor
        processor = get_human_parsing_processor()
        _segmentation_model = processor.model
    return _segmentation_model


def get_transform(image_size):
    """이미지 크기에 따른 변환 정의"""
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def segment_image(image_path: str) -> Dict[str, Any]:
    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "segmentation")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(base_name)
    segmentation_path = os.path.join(output_dir, f"{file_name}_segmentation{file_ext}")
    mask_path = os.path.join(output_dir, f"{file_name}_mask.png")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 인체 파싱 프로세서 사용
    try:
        processor = get_human_parsing_processor()

        # 세그멘테이션 처리 및 원본 크기로 변환
        _, _, orig_mask, overlay_original = processor.process_and_segment(
            image_rgb,
            canvas_size=None  # 원본 이미지 크기 사용
        )

        if orig_mask is None:
            print("[WARN] 이미지에서 사람을 찾을 수 없습니다.")
            # 빈 결과 반환
            cv2.imwrite(segmentation_path, image)
            blank_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.imwrite(mask_path, blank_mask)

            return {
                'segmentation_path': segmentation_path,
                'mask_path': mask_path
            }

        # 결과 저장
        cv2.imwrite(segmentation_path, cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"[WARN] 인체 파싱 프로세서 사용 실패: {e}")
        raise  # 오류 발생시 그대로 전파

    return {
        'segmentation_path': segmentation_path,
        'mask_path': mask_path
    }


def remove_background(segmentation_result: Dict[str, Any]) -> str:
    # 세그멘테이션 결과에서 이미지와 마스크 로드
    segmentation_path = segmentation_result['segmentation_path']
    mask_path = segmentation_result['mask_path']

    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "no_background")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(segmentation_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_nobg{file_ext}")

    # 이미지와 마스크 로드
    image = cv2.imread(segmentation_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 마스크가 없는 경우 처리
    if mask is None:
        raise ValueError(f"Mask not found at path: {mask_path}")

    # 마스크 이진화 처리 (확실히 하기 위해)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # RGBA 이미지 생성 (알파 채널 추가)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 마스크 기반으로 알파 채널 설정
    rgba[:, :, 3] = mask

    # 결과 저장
    cv2.imwrite(output_path, rgba)

    return output_path


def create_agnostic_image(segmentation_path: str, keypoints_path: str) -> str:
    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "agnostic")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(segmentation_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_agnostic{file_ext}")

    # 세그멘테이션 결과 로드
    image = cv2.imread(segmentation_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 키포인트 로드 (JSON 형식 가정)
    import json
    with open(keypoints_path, 'r') as f:
        keypoints = json.load(f)

    # 인체 파싱 프로세서 사용
    try:
        processor = get_human_parsing_processor()

        # 세그멘테이션 처리 (원본 크기 그대로)
        _, _, orig_mask, _ = processor.process_and_segment(
            image_rgb,
            canvas_size=None
        )

        if orig_mask is None:
            raise ValueError("세그멘테이션 처리에 실패했습니다.")

        # Agnostic 이미지 생성 (상의 부분 제거 - 클래스 7(outer_torso)와 클래스 10(inner_torso) 영역 마스킹)
        agnostic_image = image.copy()

        # 상의 부분 마스킹 (배경색 또는 살색으로 대체)
        skin_color = np.array([203, 192, 180])  # BGR 형식의 기본 살색 값

        # 상의 마스크 생성 (inner_torso와 outer_torso 영역)
        upper_mask = np.logical_or(orig_mask == 7, orig_mask == 10)  # outer_torso=7, inner_torso=10

        # 상의 부분을 살색으로 대체
        agnostic_image[upper_mask] = skin_color

    except Exception as e:
        print(f"[WARN] 인체 파싱 프로세서 사용 실패: {e}")
        raise  # 오류 발생시 그대로 전파

    # 결과 저장
    cv2.imwrite(output_path, agnostic_image)

    return output_path