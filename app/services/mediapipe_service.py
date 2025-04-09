import os
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Any

from app.core.config import get_settings

settings = get_settings()

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image_path: str) -> Dict[str, Any]:
    """
    MediaPipe를 사용하여 이미지에서 키포인트 추출 (Class 17)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Pose 모델 실행
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)
    
    # 키포인트 추출 및 결과 반환
    if not results.pose_landmarks:
        raise ValueError("No pose landmarks detected in the image")
    
    # 키포인트 정보 추출
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append({
            'x': landmark.x,
            'y': landmark.y, 
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    
    # 시각화 (디버깅용)
    visualize_keypoints(image_path, results)
    
    return {
        'keypoints': keypoints,
        'has_segmentation_mask': results.segmentation_mask is not None,
    }

def visualize_keypoints(image_path: str, results) -> str:
    """
    키포인트 시각화 및 저장
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 키포인트 그리기
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    # 시각화 결과 저장
    output_dir = os.path.join(settings.OUTPUT_DIR, "keypoints")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{file_name}_keypoints{file_ext}")
    
    cv2.imwrite(output_path, annotated_image)
    
    return output_path