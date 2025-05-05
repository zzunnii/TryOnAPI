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
    
    # 원본 이미지 크기 저장
    orig_h, orig_w = image_rgb.shape[:2]
    print(f"[INFO] 세그멘테이션 원본 이미지 크기: {orig_w}x{orig_h}")

    # 인체 파싱 프로세서 사용
    try:
        processor = get_human_parsing_processor()

        # 세그멘테이션 처리 - 320x640으로 처리 후 원본 크기로 변환됨
        _, _, orig_mask, overlay_original = processor.process_and_segment(
            image_rgb,
            canvas_size=None  # None으로 지정하여 processor.py의 로직에 따라 320x640 사용
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


def create_agnostic_image(segmentation_path: str, keypoints_path: str, category: str = "upper") -> str:
    """
    카테고리에 따라 의류 영역이 제거된 agnostic 이미지 생성
    
    Args:
        segmentation_path: 세그멘테이션 결과 이미지 경로
        keypoints_path: 키포인트 JSON 파일 경로
        category: 의류 카테고리 (upper, lower, dress, outer)
        
    Returns:
        str: agnostic 이미지 저장 경로
    """
    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "agnostic")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(segmentation_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_{category}_agnostic{file_ext}")
    
    # 마스크 결과 경로 (디버깅용)
    mask_path = os.path.join(output_dir, f"{file_name}_{category}_mask.png")

    print(f"[INFO] 의류 카테고리 '{category}'에 대한 agnostic 이미지 생성 시작")
    
    # 세그멘테이션 결과 로드
    image = cv2.imread(segmentation_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 키포인트 로드 (JSON 형식 가정)
    import json
    try:
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
        print(f"[INFO] 키포인트 로드 완료: {len(keypoints.get('pose_landmarks', []))} 포인트")
    except Exception as e:
        print(f"[WARN] 키포인트 로드 실패: {e}")
        keypoints = {"pose_landmarks": []}

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

        # Agnostic 이미지 생성
        agnostic_image = image.copy()
        
        # 기본 살색 값 (BGR)
        skin_color = np.array([203, 192, 180])
        
        # 카테고리별 마스크 생성
        clothing_mask = np.zeros_like(orig_mask, dtype=bool)
        
        # 카테고리별 마스킹 영역 정의
        if category.lower() == "upper":
            # 상의: 목, 상의 내부, 상의 외부, 소매 부분
            # 3: neck, 7: outer_torso, 10: inner_torso, 5,6: outer_sleeves, 8,9: inner_sleeves
            clothing_mask = np.logical_or.reduce([
                orig_mask == 7,  # outer_torso
                orig_mask == 10, # inner_torso
                orig_mask == 5,  # outer_rsleeve
                orig_mask == 6,  # outer_lsleeve
                orig_mask == 8,  # inner_rsleeve
                orig_mask == 9,  # inner_lsleeve
            ])
            print("[INFO] 상의 마스크 생성 완료")
            
        elif category.lower() == "lower":
            # 하의: 바지, 스커트, 다리 부분
            # 11: pants_hip, 12,13: pants sleeves, 14: skirt
            clothing_mask = np.logical_or.reduce([
                orig_mask == 11, # pants_hip
                orig_mask == 12, # pants_rsleeve
                orig_mask == 13, # pants_lsleeve
                orig_mask == 14, # skirt
            ])
            print("[INFO] 하의 마스크 생성 완료")
            
        elif category.lower() == "dress":
            # 드레스: 상의 + 하의 전체
            clothing_mask = np.logical_or.reduce([
                # 상의 부분
                orig_mask == 7,  # outer_torso
                orig_mask == 10, # inner_torso
                orig_mask == 5,  # outer_rsleeve
                orig_mask == 6,  # outer_lsleeve
                orig_mask == 8,  # inner_rsleeve
                orig_mask == 9,  # inner_lsleeve,
                # 하의 부분
                orig_mask == 11, # pants_hip
                orig_mask == 12, # pants_rsleeve
                orig_mask == 13, # pants_lsleeve
                orig_mask == 14, # skirt
            ])
            print("[INFO] 드레스 마스크 생성 완료")
            
        elif category.lower() == "outer":
            # 아우터: 겉옷, 소매 (안쪽 상의 포함 가능)
            clothing_mask = np.logical_or.reduce([
                orig_mask == 7,  # outer_torso
                orig_mask == 5,  # outer_rsleeve
                orig_mask == 6,  # outer_lsleeve
            ])
            print("[INFO] 아우터 마스크 생성 완료")
            
        # 키포인트를 사용하여 마스크 개선 (옵션)
        if keypoints.get('pose_landmarks', []):
            print("[INFO] 키포인트를 사용하여 마스크 개선 중...")
            clothing_mask = _enhance_mask_with_keypoints(
                clothing_mask, 
                keypoints['pose_landmarks'], 
                image.shape[:2], 
                category
            )
        
        # 마스크 스무딩 (경계 부분 부드럽게)
        kernel = np.ones((5, 5), np.uint8)
        clothing_mask_smoothed = cv2.dilate(clothing_mask.astype(np.uint8), kernel, iterations=1)
        clothing_mask_smoothed = cv2.morphologyEx(clothing_mask_smoothed, cv2.MORPH_CLOSE, kernel)
        clothing_mask_smoothed = cv2.GaussianBlur(clothing_mask_smoothed.astype(np.float32), (7, 7), 0) > 0.5
        
        # 마스크 저장 (디버깅용)
        cv2.imwrite(mask_path, clothing_mask_smoothed.astype(np.uint8) * 255)
        print(f"[INFO] 의류 마스크 저장됨: {mask_path}")
        
        # 의류 영역을 살색으로 대체
        agnostic_image[clothing_mask_smoothed] = skin_color
        
        # 추가: 자연스러운 살색 효과를 위한 블러 처리
        mask_region = clothing_mask_smoothed.astype(np.uint8) * 255
        mask_edge = cv2.dilate(mask_region, kernel) - cv2.erode(mask_region, kernel)
        agnostic_blurred = cv2.GaussianBlur(agnostic_image, (15, 15), 0)
        
        # 경계 영역에 블러 적용
        edge_pixels = mask_edge > 0
        if np.any(edge_pixels):
            agnostic_image[edge_pixels] = agnostic_blurred[edge_pixels]
        
        print(f"[INFO] Agnostic 이미지 생성 완료: 카테고리 '{category}'")

    except Exception as e:
        print(f"[WARN] Agnostic 이미지 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise  # 오류 발생시 그대로 전파

    # 결과 저장
    cv2.imwrite(output_path, agnostic_image)
    print(f"[INFO] Agnostic 이미지 저장됨: {output_path}")

    return output_path


def _enhance_mask_with_keypoints(mask, keypoints, image_shape, category):
    """
    키포인트를 사용하여 의류 마스크 개선
    
    Args:
        mask: 기존 마스크 (boolean 배열)
        keypoints: 키포인트 리스트
        image_shape: 이미지 크기 (height, width)
        category: 의류 카테고리
        
    Returns:
        numpy.ndarray: 개선된 마스크 (boolean 배열)
    """
    height, width = image_shape
    enhanced_mask = mask.copy()
    
    # 키포인트가 없으면 원본 마스크 반환
    if not keypoints:
        return enhanced_mask
    
    try:
        # 주요 키포인트 인덱스 매핑
        idx_map = {
            "nose": 0,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28
        }
        
        # 키포인트 좌표 추출
        kp_coords = {}
        for idx_name, idx in idx_map.items():
            if idx < len(keypoints):
                kp = keypoints[idx]
                x_pixel = int(kp.get("pixel_x", kp.get("x", 0) * width))
                y_pixel = int(kp.get("pixel_y", kp.get("y", 0) * height))
                kp_coords[idx_name] = (x_pixel, y_pixel)
        
        # PIL로 마스크 생성 (더 자연스러운 곡선 그리기 위해)
        from PIL import Image, ImageDraw
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # 카테고리별 마스크 생성
        if category.lower() == "upper" or category.lower() == "outer":
            # 상의/아우터: 어깨-팔꿈치-손목-허리 영역
            if all(k in kp_coords for k in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip"]):
                # 상체 다각형 그리기
                torso_poly = [
                    kp_coords["left_shoulder"],
                    kp_coords["right_shoulder"],
                    kp_coords["right_hip"],
                    kp_coords["left_hip"]
                ]
                draw.polygon(torso_poly, fill=255)
                
                # 팔 부분 그리기
                if "left_elbow" in kp_coords:
                    line_width = int(width * 0.03)  # 3% 너비
                    draw.line(
                        [kp_coords["left_shoulder"], kp_coords["left_elbow"]],
                        fill=255, width=line_width
                    )
                    if "left_wrist" in kp_coords:
                        draw.line(
                            [kp_coords["left_elbow"], kp_coords["left_wrist"]],
                            fill=255, width=line_width
                        )
                
                if "right_elbow" in kp_coords:
                    line_width = int(width * 0.03)
                    draw.line(
                        [kp_coords["right_shoulder"], kp_coords["right_elbow"]],
                        fill=255, width=line_width
                    )
                    if "right_wrist" in kp_coords:
                        draw.line(
                            [kp_coords["right_elbow"], kp_coords["right_wrist"]],
                            fill=255, width=line_width
                        )
        
        elif category.lower() == "lower":
            # 하의: 허리-무릎-발목 영역
            if all(k in kp_coords for k in ["left_hip", "right_hip", "left_knee", "right_knee"]):
                # 하의 상단 다각형
                lower_poly = [
                    kp_coords["left_hip"],
                    kp_coords["right_hip"],
                    kp_coords["right_knee"],
                    kp_coords["left_knee"]
                ]
                draw.polygon(lower_poly, fill=255)
                
                # 다리 부분 그리기
                if "left_knee" in kp_coords and "left_ankle" in kp_coords:
                    line_width = int(width * 0.04)
                    draw.line(
                        [kp_coords["left_knee"], kp_coords["left_ankle"]],
                        fill=255, width=line_width
                    )
                
                if "right_knee" in kp_coords and "right_ankle" in kp_coords:
                    line_width = int(width * 0.04)
                    draw.line(
                        [kp_coords["right_knee"], kp_coords["right_ankle"]],
                        fill=255, width=line_width
                    )
                    
                # 스커트 형태로 확장 (옵션)
                if "left_knee" in kp_coords and "right_knee" in kp_coords:
                    mid_y = (kp_coords["left_knee"][1] + kp_coords["right_knee"][1]) // 2
                    skirt_width = int(abs(kp_coords["right_hip"][0] - kp_coords["left_hip"][0]) * 1.2)
                    skirt_left = (kp_coords["left_knee"][0] - int(skirt_width * 0.3), mid_y + int(height * 0.05))
                    skirt_right = (kp_coords["right_knee"][0] + int(skirt_width * 0.3), mid_y + int(height * 0.05))
                    
                    skirt_poly = [
                        kp_coords["left_knee"],
                        kp_coords["right_knee"],
                        skirt_right,
                        skirt_left
                    ]
                    draw.polygon(skirt_poly, fill=255)
        
        elif category.lower() == "dress":
            # 드레스: 상의+하의 통합
            if all(k in kp_coords for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
                # 상체 다각형
                upper_poly = [
                    kp_coords["left_shoulder"],
                    kp_coords["right_shoulder"],
                    kp_coords["right_hip"],
                    kp_coords["left_hip"]
                ]
                draw.polygon(upper_poly, fill=255)
                
                # 팔 부분
                if "left_elbow" in kp_coords:
                    line_width = int(width * 0.03)
                    draw.line(
                        [kp_coords["left_shoulder"], kp_coords["left_elbow"]],
                        fill=255, width=line_width
                    )
                    if "left_wrist" in kp_coords:
                        draw.line(
                            [kp_coords["left_elbow"], kp_coords["left_wrist"]],
                            fill=255, width=line_width
                        )
                
                if "right_elbow" in kp_coords:
                    line_width = int(width * 0.03)
                    draw.line(
                        [kp_coords["right_shoulder"], kp_coords["right_elbow"]],
                        fill=255, width=line_width
                    )
                    if "right_wrist" in kp_coords:
                        draw.line(
                            [kp_coords["right_elbow"], kp_coords["right_wrist"]],
                            fill=255, width=line_width
                        )
                
                # 드레스 치마 부분 (A라인)
                skirt_bottom = int(height * 0.85)
                skirt_width = int(abs(kp_coords["right_hip"][0] - kp_coords["left_hip"][0]) * 1.8)
                skirt_left = (kp_coords["left_hip"][0] - int(skirt_width * 0.3), skirt_bottom)
                skirt_right = (kp_coords["right_hip"][0] + int(skirt_width * 0.3), skirt_bottom)
                
                skirt_poly = [
                    kp_coords["left_hip"],
                    kp_coords["right_hip"],
                    skirt_right,
                    skirt_left
                ]
                draw.polygon(skirt_poly, fill=255)
        
        # PIL 마스크를 numpy 배열로 변환
        keypoint_mask = np.array(mask_img) > 0
        
        # 원본 마스크와 키포인트 마스크 통합
        enhanced_mask = np.logical_or(enhanced_mask, keypoint_mask)
        
        # 마스크 스무딩 (경계 부분 부드럽게)
        kernel = np.ones((5, 5), np.uint8)
        enhanced_mask = cv2.dilate(enhanced_mask.astype(np.uint8), kernel, iterations=1)
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
        enhanced_mask = enhanced_mask > 0
        
    except Exception as e:
        print(f"[WARN] 키포인트 기반 마스크 개선 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return enhanced_mask