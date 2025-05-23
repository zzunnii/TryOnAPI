import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch

# Import settings to get model paths
from api.core.config import get_settings
settings = get_settings()

class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        # 설정 가져오기
        from api.core.config import get_settings
        settings = get_settings()

        self.model_path = settings.HUMAN_PARSING_ATR_MODEL
        atr_model_path = settings.HUMAN_PARSING_ATR_MODEL
        lip_model_path = settings.HUMAN_PARSING_LIP_MODEL

        print(f"Loading human parsing model from: {atr_model_path}")
        print(f"Loading lip parsing model from: {lip_model_path}")
        
        self.session = ort.InferenceSession(atr_model_path,
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(lip_model_path,
                                                sess_options=session_options, providers=['CPUExecutionProvider'])
        

    def __call__(self, input_image):
        # torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
