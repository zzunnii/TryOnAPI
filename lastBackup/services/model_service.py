"""
IDM-VTON 모델 서비스

모델 로딩 및 가상 피팅을 위한 핵심 서비스
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 앱 설정 가져오기
from api.core.config import get_settings
settings = get_settings()

# 유틸리티
from api.utils.image_utils import save_image, resize_image

# 싱글톤 패턴 구현
class ModelService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        from api.core.config import get_settings
        settings = get_settings()

        # 모델별 이미지 크기 설정
        self.densepose_image_size = settings.DENSEPOSE_IMAGE_SIZE
        self.idm_vton_image_size = settings.IDM_VTON_IMAGE_SIZE
        # 기존 코드에서 사용하던 image_size는 기본값으로 설정
        self.image_size = settings.DEFAULT_IMAGE_SIZE

        # 초기화 플래그
        self._initialized = True
        self._models_loaded = False
        
        # 장치 설정
        self.device = settings.DEVICE

        # 모델 컴포넌트
        self.unet = None
        self.unet_encoder = None
        self.vae = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.image_encoder = None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.scheduler = None
        self.pipe = None
        self.parsing_model = None
        self.openpose_model = None


    def load_models(self):
        """모델 로드 함수"""
        try:
            from api.core.config import get_settings
            settings = get_settings()
            # 필요한 라이브러리 임포트
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer
            )
            from diffusers import DDPMScheduler, AutoencoderKL
            
            # IDM-VTON 모델 컴포넌트 가져오기
            import sys
            # diffusionModels 모듈 경로 추가
            sys.path.append(os.path.join(settings.BASE_DIR, "diffusionModels"))
            
            from api.diffusionModels.models.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
            from api.diffusionModels.models.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from api.diffusionModels.models.unet_hacked_tryon import UNet2DConditionModel
            
            # 경로 설정
            base_model_path = settings.MODEL_CHECKPOINT_PATH

            print(f"모델 로드 시작: {base_model_path}")
            
            # 필요한 모델 컴포넌트 로드
            self.unet = UNet2DConditionModel.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="unet",
                torch_dtype=torch.float16,
            )
            self.unet.requires_grad_(False)
            
            self.tokenizer_one = AutoTokenizer.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )
            
            self.tokenizer_two = AutoTokenizer.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False,
            )
            
            self.scheduler = DDPMScheduler.from_pretrained(settings.IDM_VTON_MODEL_DIR, subfolder="scheduler")
            
            self.text_encoder_one = CLIPTextModel.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="text_encoder",
                torch_dtype=torch.float16,
            )
            
            self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16,
            )
            
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="image_encoder",
                torch_dtype=torch.float16,
            )
            
            self.vae = AutoencoderKL.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="vae",
                torch_dtype=torch.float16,
            )
            
            self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                subfolder="unet_encoder",
                torch_dtype=torch.float16,
            )
            
            # 모델을 GPU로 이동
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                self.unet = self.unet.to(self.device)
                self.text_encoder_one = self.text_encoder_one.to(self.device)
                self.text_encoder_two = self.text_encoder_two.to(self.device)
                self.image_encoder = self.image_encoder.to(self.device)
                self.vae = self.vae.to(self.device)
                self.unet_encoder = self.unet_encoder.to(self.device)
            
            # 파이프라인 생성
            self.pipe = TryonPipeline.from_pretrained(
                settings.IDM_VTON_MODEL_DIR,
                unet=self.unet,
                vae=self.vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=self.text_encoder_one,
                text_encoder_2=self.text_encoder_two,
                tokenizer=self.tokenizer_one,
                tokenizer_2=self.tokenizer_two,
                scheduler=self.scheduler,
                image_encoder=self.image_encoder,
                torch_dtype=torch.float16,
                local_files_only=True
            )
            self.pipe.unet_encoder = self.unet_encoder
            
            # 경량화된 평가모드 설정
            self.unet_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            
            # 선택적으로 추가 모델 로드 (Parsing, OpenPose 등)
            # (필요시 구현)
            
            self._models_loaded = True
            print("모델 로드 완료.")
        
        except Exception as e:
            print(f"모델 로드 중 오류: {str(e)}")
            traceback.print_exc()
            self._models_loaded = False
            raise
    
    def generate_tryon_image(
        self,
        person_image_path: str,
        garment_image_path: str,
        mask_image_path: str,
        pose_image_path: str,
        garment_description: str,
        category: str,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        guidance_scale: float = 2.0,
        output_path: Optional[str] = None
    ) -> dict:
        """
        가상 피팅 이미지 생성 함수
        
        Args:
            person_image_path: 사용자 이미지 경로
            garment_image_path: 의류 이미지 경로
            mask_image_path: 마스크 이미지 경로
            pose_image_path: 포즈 이미지 경로 (DensePose)
            garment_description: 의류 설명 텍스트
            category: 의류 카테고리 (upper, lower, dress, outer)
            num_inference_steps: 추론 스텝 수
            seed: 랜덤 시드 (None이면 랜덤 생성)
            guidance_scale: 가이던스 스케일
            output_path: 결과 저장 경로 (None이면 자동 생성)
            
        Returns:
            dict: 처리 결과 정보
        """
        if not self._models_loaded:
            try:
                self.load_models()
            except Exception as e:
                return {"status": "error", "message": f"모델 로드 오류: {str(e)}"}
        
        try:
            # GPU로 모델 이동
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                self.pipe.to(self.device)
                self.pipe.unet_encoder.to(self.device)
            
            print(f"가상 피팅 처리 시작...")
            
            # 이미지 로드 및 전처리
            from torchvision import transforms
            from torchvision.transforms.functional import to_pil_image
            
            tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
            # 원본 크기 정보 가져오기 함수
            def get_original_size(image_path):
                """이미지의 원본 크기 정보 가져오기"""
                # 경로에서 ID 추출 (예: /path/to/parsing_temp_1234abcd.jpg -> 1234abcd)
                import re
                import os
                import json
                
                # 파일명에서 ID 추출
                filename = os.path.basename(image_path)
                match = re.search(r'_([a-f0-9\-]+)\.', filename)
                if not match:
                    return None  # ID를 찾을 수 없음
                
                file_id = match.group(1)
                
                # 각 전처리 결과 디렉토리에서 size_info 파일 검색
                directories = [
                    os.path.join(settings.HUMAN_PARSING_DIR),
                    os.path.join(settings.OUTPUT_DIR, "densepose"),
                    os.path.join(settings.OUTPUT_DIR, "pose"),
                    os.path.join(settings.OUTPUT_DIR, "masks")
                ]
                
                for directory in directories:
                    size_info_path = os.path.join(directory, f"{file_id}_size_info.json")
                    if os.path.exists(size_info_path):
                        with open(size_info_path, 'r') as f:
                            metadata = json.load(f)
                            return (metadata.get('original_width'), metadata.get('original_height'))
                
                # 메타데이터를 찾을 수 없는 경우 None 반환
                return None
            
            # 이미지 로드
            person_img_original = Image.open(person_image_path).convert("RGB")
            garment_img_original = Image.open(garment_image_path).convert("RGB")
            pose_img_original = Image.open(pose_image_path).convert("RGB")
            mask_image_original = Image.open(mask_image_path).convert("L")
            
            # 원본 크기 정보 가져오기
            person_original_size = get_original_size(person_image_path)
            garment_original_size = get_original_size(garment_image_path)
            pose_original_size = get_original_size(pose_image_path)
            mask_original_size = get_original_size(mask_image_path)
            
            # 원본 크기로 복원 (크기 정보가 있는 경우에만)
            from api.utils.image_utils import resize_to_original
            
            if person_original_size:
                person_img_restored = resize_to_original(person_img_original, person_original_size)
            else:
                person_img_restored = person_img_original
            
            if garment_original_size:
                garment_img_restored = resize_to_original(garment_img_original, garment_original_size)
            else:
                garment_img_restored = garment_img_original
            
            if pose_original_size:
                pose_img_restored = resize_to_original(pose_img_original, pose_original_size)
            else:
                pose_img_restored = pose_img_original
            
            if mask_original_size:
                mask_image_restored = resize_to_original(mask_image_original, mask_original_size)
            else:
                mask_image_restored = mask_image_original
            
            # 최종 모델 입력을 위한 리사이징
            target_size = (768, 1024)
            person_img = resize_image(person_img_restored, target_size)
            garment_img = resize_image(garment_img_restored, target_size)
            pose_img = resize_image(pose_img_restored, target_size)
            mask_image = resize_image(mask_image_restored, target_size)
            
            # 텐서 변환
            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
            garment_tensor = tensor_transform(garment_img).unsqueeze(0).to(self.device, torch.float16)
            
            # 텍스트 임베딩 생성
            prompt = f"model is wearing {garment_description}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                prompt_cloth = f"a photo of {garment_description}"
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = self.pipe.encode_prompt(
                        prompt_cloth,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )
                
                # 랜덤 시드 설정
                generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
                
                # 파이프라인 실행
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_tensor.to(self.device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=garment_tensor.to(self.device, torch.float16),
                    mask_image=mask_image,
                    image=person_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garment_img,
                    guidance_scale=guidance_scale,
                )[0]
            
            # 결과 저장
            if output_path is None:
                import uuid
                output_path = os.path.join(settings.TRYON_DIR, f"{uuid.uuid4()}.jpg")
            
            # 결과 이미지 저장
            images[0].save(output_path)
            
            print(f"가상 피팅 완료: {output_path}")
            
            # 결과 반환
            return {
                "status": "success",
                "message": "가상 피팅 완료",
                "output_path": output_path,
                "category": category
            }
            
        except Exception as e:
            print(f"가상 피팅 오류: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"가상 피팅 처리 중 오류: {str(e)}"}
    
    def get_device_info(self) -> Dict[str, Any]:
        """시스템 디바이스 정보 반환"""
        info = {
            "device": self.device,
            "torch_version": torch.__version__,
            "models_loaded": self._models_loaded,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "cuda_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
            })
        else:
            info["cuda_available"] = False
        
        return info


def get_model_service() -> ModelService:
    """
    ModelService 싱글톤 인스턴스 반환
    """
    return ModelService()