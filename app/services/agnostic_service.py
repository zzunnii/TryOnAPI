"""
Agnostic 이미지 생성 서비스
가상 착용을 위한 agnostic 이미지 및 마스크 생성 서비스
"""
import os
import uuid
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import traceback

from app.core.config import get_settings
from app.utils.agnostic_utils import create_agnostic_image

settings = get_settings()

class AgnosticService:
    """
    Agnostic 이미지 생성 서비스 클래스
    - 의류 영역이 마스킹된 이미지(agnostic) 생성
    - 의류 영역 마스크 생성
    """
    
    def __init__(self):
        """서비스 초기화"""
        self.output_dir = os.path.join(settings.OUTPUT_DIR, "agnostic")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_agnostic(
        self,
        image_path: str,
        parsing_json_path: str,
        keypoints_json_path: str,
        category: str = "long_dress"
    ) -> Dict[str, Any]:
        """
        카테고리에 맞는 agnostic 이미지 및 마스크 생성
        
        Args:
            image_path: 사용자 이미지 경로
            parsing_json_path: 인체 파싱 결과 JSON 파일 경로
            keypoints_json_path: 미디어파이프 키포인트 JSON 파일 경로
            category: 의류 카테고리 (upper, lower, dress, long_dress, skirt 등)
            
        Returns:
            Dict[str, Any]: 결과 정보 (agnostic_path, mask_path, agnostic_id 등)
        """
        try:
            print(f"\n=== Agnostic 이미지 생성 시작: {category} 카테고리 ===")
            print(f"사용자 이미지: {image_path}")
            print(f"파싱 결과: {parsing_json_path}")
            print(f"키포인트 정보: {keypoints_json_path}")
            
            # 결과 ID 및 파일 경로 설정
            result_id = str(uuid.uuid4())
            agnostic_path = os.path.join(self.output_dir, f"{result_id}_agnostic.jpg")
            mask_path = os.path.join(self.output_dir, f"{result_id}_mask.png")
            
            # Agnostic 이미지 및 마스크 생성
            agnostic_image, mask = create_agnostic_image(
                image_path=image_path,
                parsing_json_path=parsing_json_path,
                keypoints_json_path=keypoints_json_path,
                category=category,
                output_path=agnostic_path
            )
            
            # 생성 확인
            if not os.path.exists(agnostic_path):
                print("Agnostic 이미지 생성 실패! 직접 저장 시도...")
                agnostic_image.save(agnostic_path)
            
            if not os.path.exists(mask_path):
                print("마스크 이미지 생성 실패! 직접 저장 시도...")
                cv2.imwrite(mask_path, mask * 255)
            
            # 결과 반환
            result = {
                "status": "success",
                "agnostic_id": result_id,
                "agnostic_path": agnostic_path,
                "mask_path": mask_path,
                "category": category,
                "width": agnostic_image.width,
                "height": agnostic_image.height
            }
            
            print(f"Agnostic 이미지 생성 완료!")
            return result
            
        except Exception as e:
            print(f"Agnostic 이미지 생성 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            
            # 오류 정보 반환
            return {
                "status": "error",
                "message": str(e),
                "category": category
            }
    
    def get_agnostic_by_id(self, agnostic_id: str) -> Optional[Dict[str, Any]]:
        """
        ID로 기존 agnostic 이미지 및 마스크 정보 조회
        
        Args:
            agnostic_id: Agnostic 이미지 ID
            
        Returns:
            Optional[Dict[str, Any]]: 결과 정보 (존재하지 않으면 None)
        """
        agnostic_path = os.path.join(self.output_dir, f"{agnostic_id}_agnostic.jpg")
        mask_path = os.path.join(self.output_dir, f"{agnostic_id}_mask.png")
        
        if os.path.exists(agnostic_path) and os.path.exists(mask_path):
            # 이미지 크기 정보 가져오기
            agnostic_image = Image.open(agnostic_path)
            
            return {
                "status": "success",
                "agnostic_id": agnostic_id,
                "agnostic_path": agnostic_path,
                "mask_path": mask_path,
                "width": agnostic_image.width,
                "height": agnostic_image.height
            }
        
        return None

# 싱글톤 인스턴스
_agnostic_service = None

def get_agnostic_service():
    """Agnostic 서비스 싱글톤 인스턴스 반환"""
    global _agnostic_service
    if _agnostic_service is None:
        _agnostic_service = AgnosticService()
    return _agnostic_service