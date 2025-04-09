from pydantic import BaseSettings
from typing import Optional, List
from functools import lru_cache

class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "TryOn API"
    
    # 모델 관련 설정
    MEDIAPIPE_MODEL_PATH: str = "models/mediapipe"
    SEGMENTATION_MODEL_PATH: str = "models/segmentation"
    DIFFUSION_MODEL_256_PATH: str = "models/diffusion_256"
    DIFFUSION_MODEL_512_PATH: str = "models/diffusion_512"
    DIFFUSION_MODEL_1024_PATH: str = "models/diffusion_1024"
    
    # 이미지 처리 관련 설정
    TEMP_DIR: str = "temp"
    OUTPUT_DIR: str = "output"
    MAX_IMAGE_SIZE: int = 1024
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache
def get_settings():
    return Settings()