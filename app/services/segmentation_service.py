import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms

# Human Parsing 모듈 경로 추가
sys.path.insert(0, '/Users/iseongjun1/HumanParsing')

# Human Parsing 모델 import
from ..models.parsingModels.parsing_model import ParsingModel

from app.core.config import get_settings

settings = get_settings()

# 세그멘테이션을 위한 전처리 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 로드를 위한 함수
def load_segmentation_model(device: torch.device = None) -> ParsingModel:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 설정
    model = ParsingModel(
        num_classes=21,  # 사람 파싱에 필요한 클래스 수 (배경 포함)
        backbone_pretrained=False,  # pretrained 모델 불필요
        fpn_channels=256,
        decoder_channels=512
    )
    
    # 체크포인트 경로
    checkpoint_path = os.path.join(settings.SEGMENTATION_MODEL_PATH, '../models/human_parsing_model.pth')  # 여기에 model 경로를 추가하세요
    
    # 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model


# 세그멘테이션 모델 캐시 (필요할 때만 로드하도록)
_segmentation_model = None


def get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is None:
        _segmentation_model = load_segmentation_model()
    return _segmentation_model


def segment_image(image_path: str) -> Dict[str, Any]:
    """
    이미지 세그멘테이션 수행 (Class 20)
    Human Parsing 모델을 사용하여 사람 이미지 세그멘테이션 수행
    """
    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "segmentation")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(base_name)
    segmentation_path = os.path.join(output_dir, f"{file_name}_segmentation{file_ext}")
    mask_path = os.path.join(output_dir, f"{file_name}_mask.png")
    
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 이미지 전처리
    image_tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    
    # 모델 로드
    model = get_segmentation_model()
    device = model.device
    image_tensor = image_tensor.to(device)
    
    # 세그멘테이션 수행
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # (H, W)
    
    # 클래스 20 (상의 영역) 마스크 추출
    upper_body_mask = (predictions == 20).astype(np.uint8) * 255
    
    # 세그멘테이션 결과 시각화 (컬러 맵 적용)
    colormap = np.zeros((21, 3), dtype=np.uint8)
    colormap[20] = [255, 0, 0]  # 상의 영역은 빨간색으로 시각화
    
    # 원본 이미지에 세그멘테이션 결과 오버레이
    original_image = np.array(image)
    segmentation_overlay = original_image.copy()
    
    # 세그멘테이션 결과 적용
    for class_idx in range(1, 21):  # 0은 배경이므로 제외
        mask = (predictions == class_idx)
        if np.any(mask):
            color = colormap[class_idx]
            segmentation_overlay[mask] = original_image[mask] * 0.7 + color * 0.3
    
    # 결과 저장
    cv2.imwrite(segmentation_path, cv2.cvtColor(segmentation_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, upper_body_mask)
    
    return {
        'segmentation_path': segmentation_path,
        'mask_path': mask_path
    }

def remove_background(segmentation_result: Dict[str, Any]) -> str:
    """
    세그멘테이션 결과를 사용하여 배경 제거
    """
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
    """
    세그멘테이션 결과와 키포인트를 사용하여 Agnostic 이미지 생성
    Agnostic 이미지: 의상을 가상 착용할 때 옷이 입혀질 부분을 제거한 이미지
    """
    # 결과 저장 경로
    output_dir = os.path.join(settings.OUTPUT_DIR, "agnostic")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(segmentation_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_agnostic{file_ext}")
    
    # 세그멘테이션 결과 로드
    image = cv2.imread(segmentation_path)
    
    # 키포인트 로드 (JSON 형식 가정)
    import json
    with open(keypoints_path, 'r') as f:
        keypoints = json.load(f)
    
    # 모델 다시 로드하여 전체 세그멘테이션 맵 얻기
    model = get_segmentation_model()
    
    # 이미지 전처리
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tensor = transform(image_pil).unsqueeze(0)
    
    # 세그멘테이션 다시 수행
    with torch.no_grad():
        device = model.device
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    # Agnostic 이미지 생성 (상의 부분 제거 - 클래스 5(상의)와 클래스 6(드레스) 영역 마스킹)
    agnostic_image = image.copy()
    
    # 상의/드레스 부분 마스킹 (배경색 또는 살색으로 대체)
    # 일반적으로 살색 값은 아래와 같이 정의할 수 있음
    skin_color = np.array([203, 192, 180])  # BGR 형식의 기본 살색 값
    
    # 상의 마스크 생성 (상의 영역은 클래스 5, 드레스는 클래스 6으로 가정)
    upper_mask = np.logical_or(predictions == 5, predictions == 6)
    
    # 상의 부분을 살색으로 대체
    agnostic_image[upper_mask] = skin_color
    
    # 키포인트를 사용하여 자연스러운 경계 생성 (실제 구현에서는 더 정교한 방식 필요)
    # 키포인트 기반으로 경계선 생성 로직 추가...
    
    # 결과 저장
    cv2.imwrite(output_path, agnostic_image)
    
    return output_path