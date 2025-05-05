"""
MediaPipe 서비스 - 인체 키포인트 추출
"""

import os
import cv2
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional

# mediapipe 임포트 - 실제 사용 시 설치 필요
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe를 사용할 수 없습니다. pip install mediapipe로 설치하세요.")

from app.core.config import get_settings

settings = get_settings()

class MediaPipeService:
    """MediaPipe를 사용한 인체 키포인트 추출 서비스"""
    
    def __init__(self):
        """초기화 함수 - MediaPipe 모델 로드"""
        if not MEDIAPIPE_AVAILABLE:
            print("경고: MediaPipe를 사용할 수 없습니다. 가짜 키포인트가 반환됩니다.")
            self.mp_pose = None
            self.mp_drawing = None
            self.pose = None
        else:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # MediaPipe Pose 모델 초기화
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
    
    def extract_keypoints(self, image_path: str) -> Dict[str, Any]:
        """
        이미지에서 인체 키포인트 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict: 키포인트 정보
        """
        # MediaPipe를 사용할 수 없는 경우 가짜 데이터 반환
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return self._generate_mock_keypoints()
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            
            # BGR -> RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Pose 처리
            results = self.pose.process(image_rgb)
            
            # 결과 변환
            keypoints_data = self._convert_results_to_dict(results, image.shape)
            
            return keypoints_data
            
        except Exception as e:
            print(f"MediaPipe 키포인트 추출 오류: {str(e)}")
            return self._generate_mock_keypoints()
    
    def _convert_results_to_dict(self, results, image_shape) -> Dict[str, Any]:
        """MediaPipe 결과를 사용하기 쉬운 딕셔너리로 변환"""
        height, width, _ = image_shape
        
        # 결과 딕셔너리 초기화
        keypoints_data = {
            "image_height": height,
            "image_width": width,
            "pose_landmarks": [],
            "pose_world_landmarks": [],
            "segmentation_mask": None
        }
        
        # Pose Landmarks 추출
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # 키포인트 좌표를 이미지 크기에 맞게 변환
                keypoints_data["pose_landmarks"].append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "pixel_x": int(landmark.x * width),
                    "pixel_y": int(landmark.y * height)
                })
        
        # World Landmarks 추출 (3D 좌표)
        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                keypoints_data["pose_world_landmarks"].append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
        
        # 세그멘테이션 마스크는 저장하지 않고 존재 여부만 표시
        if results.segmentation_mask is not None:
            keypoints_data["has_segmentation_mask"] = True
            
            # 마스크를 별도 파일로 저장하거나 바이너리 형태로 변환할 수 있음
            # 여기서는 생략
        else:
            keypoints_data["has_segmentation_mask"] = False
        
        return keypoints_data
    
    def _generate_mock_keypoints(self) -> Dict[str, Any]:
        """MediaPipe를 사용할 수 없을 때 가짜 키포인트 생성"""
        # 기본 이미지 크기 (가상)
        height, width = 1080, 720
        
        # 가짜 키포인트 33개 생성 (MediaPipe Pose 키포인트 수)
        mock_landmarks = []
        for i in range(33):
            # 랜덤 좌표 생성
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            z = np.random.uniform(-0.5, 0.5)
            visibility = np.random.uniform(0.5, 1.0)
            
            mock_landmarks.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "visibility": float(visibility),
                "pixel_x": int(x * width),
                "pixel_y": int(y * height)
            })
        
        # 가짜 월드 랜드마크도 생성
        mock_world_landmarks = []
        for i in range(33):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            visibility = np.random.uniform(0.5, 1.0)
            
            mock_world_landmarks.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "visibility": float(visibility)
            })
        
        return {
            "image_height": height,
            "image_width": width,
            "pose_landmarks": mock_landmarks,
            "pose_world_landmarks": mock_world_landmarks,
            "has_segmentation_mask": False,
            "mock_data": True
        }
    
    def visualize_keypoints(self, image_path: str, keypoints_data: Dict[str, Any], output_path: str) -> str:
        """
        키포인트를 시각화하여 이미지로 저장
        
        Args:
            image_path: 원본 이미지 경로
            keypoints_data: 키포인트 데이터
            output_path: 출력 이미지 경로
            
        Returns:
            str: 출력 이미지 경로
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            
            # MediaPipe를 사용할 수 있는 경우 직접 시각화
            if MEDIAPIPE_AVAILABLE and self.mp_drawing is not None:
                try:
                    # 키포인트 데이터를 MediaPipe 형식으로 변환
                    landmarks = self._dict_to_mp_landmarks(keypoints_data["pose_landmarks"])
                    
                    # 키포인트 그리기
                    if landmarks is not None:
                        self.mp_drawing.draw_landmarks(
                            image,
                            landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    else:
                        # MediaPipe 변환 실패 시 수동 그리기로 대체
                        print("MediaPipe 랜드마크 변환 실패, 수동 그리기로 전환합니다.")
                        self._manually_draw_keypoints(image, keypoints_data["pose_landmarks"])
                except Exception as e:
                    print(f"MediaPipe 시각화 오류, 수동 그리기로 전환: {str(e)}")
                    self._manually_draw_keypoints(image, keypoints_data["pose_landmarks"])
            else:
                # MediaPipe를 사용할 수 없는 경우 직접 그리기
                self._manually_draw_keypoints(image, keypoints_data["pose_landmarks"])
            
            # 결과 저장
            cv2.imwrite(output_path, image)
            
            return output_path
            
        except Exception as e:
            print(f"키포인트 시각화 오류: {str(e)}")
            
            # 오류 발생 시 원본 이미지를 그대로 저장
            try:
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    cv2.imwrite(output_path, image)
            except:
                pass
            
            return output_path
    
    def _dict_to_mp_landmarks(self, landmarks_dict):
        """딕셔너리를 MediaPipe 랜드마크 형식으로 변환"""
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        try:
            # PoseLandmarkList를 직접 생성하지 않고 대안 방법 사용
            import mediapipe as mp
            from mediapipe.framework.formats import landmark_pb2
            
            landmarks = landmark_pb2.NormalizedLandmarkList()
            
            for lm_dict in landmarks_dict:
                landmark = landmark_pb2.NormalizedLandmark()
                landmark.x = lm_dict["x"] 
                landmark.y = lm_dict["y"]
                landmark.z = lm_dict["z"]
                landmark.visibility = lm_dict["visibility"]
                landmarks.landmark.append(landmark)
            
            return landmarks
        except Exception as e:
            print(f"랜드마크 변환 오류: {str(e)}")
            return None
    
    def _manually_draw_keypoints(self, image, landmarks_dict):
        """MediaPipe 없이 키포인트를 수동으로 그리기"""
        height, width, _ = image.shape
        
        # 키포인트 그리기
        for lm in landmarks_dict:
            x, y = int(lm["x"] * width), int(lm["y"] * height)
            visibility = lm["visibility"]
            
            # 가시성에 따라 색상 결정
            color = (0, int(255 * visibility), int(255 * (1 - visibility)))
            
            # 원 그리기
            cv2.circle(image, (x, y), 5, color, -1)
        
        # 연결선 그리기 (MediaPipe POSE_CONNECTIONS 참고하여 필요한 연결 추가)
        # 여기서는 일부 주요 연결만 예시로 표시
        connections = [
            # 몸통
            (11, 12), (12, 24), (24, 23), (23, 11),
            # 왼팔
            (11, 13), (13, 15),
            # 오른팔
            (12, 14), (14, 16),
            # 왼다리
            (23, 25), (25, 27),
            # 오른다리
            (24, 26), (26, 28),
            # 얼굴
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)
        ]
        
        # 33개 키포인트를 모두 가지고 있는 경우에만 연결
        if len(landmarks_dict) >= 33:
            for connection in connections:
                p1, p2 = connection
                if p1 < len(landmarks_dict) and p2 < len(landmarks_dict):
                    x1 = int(landmarks_dict[p1]["x"] * width)
                    y1 = int(landmarks_dict[p1]["y"] * height)
                    x2 = int(landmarks_dict[p2]["x"] * width)
                    y2 = int(landmarks_dict[p2]["y"] * height)
                    
                    # 두 점의 가시성 평균에 따라 선 투명도 조정
                    visibility = (landmarks_dict[p1]["visibility"] + landmarks_dict[p2]["visibility"]) / 2
                    if visibility > 0.3:  # 가시성이 일정 이상일 때만 선 그리기
                        color = (0, int(255 * visibility), int(255 * (1 - visibility)))
                        cv2.line(image, (x1, y1), (x2, y2), color, 2)

# 싱글톤 인스턴스
_mediapipe_service = None

def get_mediapipe_service():
    """MediaPipe 서비스 싱글톤 인스턴스 반환"""
    global _mediapipe_service
    if _mediapipe_service is None:
        _mediapipe_service = MediaPipeService()
    return _mediapipe_service
