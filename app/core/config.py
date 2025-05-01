from pydantic import BaseModel
from typing import Optional, List, Tuple
from functools import lru_cache
import os

# BaseSettings 대신 BaseModel 사용 (Pydantic 2.x 호환)
class Settings(BaseModel):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "TryOn API"
    
    # 기본 경로
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 모델 관련 설정
    MEDIAPIPE_MODEL_PATH: str = os.path.join(BASE_DIR, "models/mediapipe")
    SEGMENTATION_MODEL_PATH: str = os.path.join(BASE_DIR, "models")
    
    # 이미지 처리 관련 설정
    TEMP_DIR: str = os.path.join(BASE_DIR, "temp")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    MAX_IMAGE_SIZE: int = 1024
    
    # Firebase 설정 (기본적으로 비활성화)
    USE_FIREBASE: bool = False
    FIREBASE_COLLECTION: str = "human_parsing"
    
    # Human Parsing 설정
    INPUT_SIZE: Tuple[int, int] = (320, 640)
    SAVE_VISUALIZATION: bool = True
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    class Config:
        # 기본 설정
        env_file = ".env"
        case_sensitive = True
        # BaseSettings 대신 BaseModel 사용 시 이 설정들은 무시될 수 있습니다

@lru_cache
def get_settings():
    return Settings()
