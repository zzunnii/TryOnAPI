"""
API 설정 모듈
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """API 설정 클래스"""

    # 앱 설정
    APP_NAME: str = "TryOnAPI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # 경로 설정
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEMP_DIR: str = os.path.join(BASE_DIR, "temp")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    HUMAN_PARSING_DIR: str = os.path.join(OUTPUT_DIR, "human_parsing")

    # 이미지 저장 디렉토리
    USER_IMAGES_DIR: str = os.path.join(OUTPUT_DIR, "user_images")
    TARGET_CLOTHING_DIR: str = os.path.join(OUTPUT_DIR, "clothing")
    TRYON_DIR: str = os.path.join(OUTPUT_DIR, "tryon")

    # 모델 체크포인트 경로
    MODEL_CHECKPOINT_PATH: str = os.path.join(BASE_DIR, "models")  # 추가된 줄

    # 모델 경로 설정 추가/수정
    BASE_MODEL_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "models")  # 상위 디렉토리의 models

    # 인체 파싱 모델 경로
    HUMAN_PARSING_MODEL_DIR: str = os.path.join(BASE_MODEL_DIR, "idm_vton", "yisol_idm_vton", "humanparsing")
    HUMAN_PARSING_ATR_MODEL: str = os.path.join(HUMAN_PARSING_MODEL_DIR, "parsing_atr.onnx")
    HUMAN_PARSING_LIP_MODEL: str = os.path.join(HUMAN_PARSING_MODEL_DIR, "parsing_lip.onnx")

    # OpenPose 모델 경로
    OPENPOSE_MODEL_DIR: str = os.path.join(BASE_MODEL_DIR, "idm_vton", "yisol_idm_vton", "openpose")
    OPENPOSE_BODY_MODEL: str = os.path.join(OPENPOSE_MODEL_DIR, "ckpts", "body_pose_model.pth")

    # IDM-VTON 모델 경로
    IDM_VTON_MODEL_DIR: str = os.path.join(BASE_MODEL_DIR, "idm_vton", "yisol_idm_vton")
    IDM_VTON_CHECKPOINT_DIR: str = os.path.join(IDM_VTON_MODEL_DIR, "checkpoints")  # 체크포인트 디렉토리가 있는 경우
    # DensePose 설정
    DENSEPOSE_API_URL: str = "http://localhost:8001"  # 외부 API 서버 URL

    # 모델별 이미지 크기 설정
    DENSEPOSE_IMAGE_SIZE: tuple = (384, 512)  # DensePose 모델용 (width, height)
    HUMAN_PARSING_IMAGE_SIZE: tuple = (384, 512)  # 인체 파싱 모델용
    IDM_VTON_IMAGE_SIZE: tuple = (768, 1024)  # IDM-VTON 모델용 (또는 실제 필요한 크기)
    DEFAULT_IMAGE_SIZE: tuple = (384, 512)  # 기본 이미지 크기

    # 장치 설정
    DEVICE: str = "cuda"  # "cuda" 또는 "cpu"

    # 카테고리 설정
    VALID_CATEGORIES: list = ["upper", "lower", "dress", "outer"]
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


@lru_cache()
def get_settings():
    """설정 인스턴스 반환 (캐시 적용)"""
    settings = Settings()

    # 필요한 디렉토리 생성
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.HUMAN_PARSING_DIR, exist_ok=True)
    os.makedirs(settings.USER_IMAGES_DIR, exist_ok=True)
    os.makedirs(settings.TARGET_CLOTHING_DIR, exist_ok=True)
    os.makedirs(settings.TRYON_DIR, exist_ok=True)

    # 모델 경로 확인 (이미 존재하는 경로는 생성하지 않고 확인만 함)
    model_paths = [
        settings.HUMAN_PARSING_ATR_MODEL,
        settings.HUMAN_PARSING_LIP_MODEL,
        settings.OPENPOSE_BODY_MODEL
    ]

    for path in model_paths:
        if not os.path.exists(path):
            print(f"경고: 모델 파일을 찾을 수 없습니다: {path}")

    return settings