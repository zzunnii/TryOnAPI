"""
IDM VTON 서비스 - 가상 피팅 처리를 위한 서비스 로직
"""
import os
from PIL import Image, ImageDraw
import torch
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
import cv2
import torch.nn.functional as F
import uuid
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import traceback

from app.core.config import get_settings

settings = get_settings()

class IDMVTONService:
    """IDM VTON 모델 서비스 클래스"""

    def __init__(self):
        """초기화 함수"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"IDM VTON 서비스 초기화, 사용 장치: {self.device}")

        # IDM VTON 모델 로드
        self.model = self._load_model()

        # 정규화 파라미터
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # 표준 이미지 크기
        self.img_size = (512, 384)  # (width, height)

        # 카테고리 매핑 정의
        self.category_mapping = {
            "upper": ["upper_body", "upper_clothes"],
            "lower": ["lower_body", "pants", "skirt"],
            "dress": ["dress"],
            "outer": ["coat", "jacket", "outer"]
        }

        # 세그멘테이션 파트 클래스 매핑
        self.segmentation_parts = {
            "background": 0,
            "hair": 1,
            "face": 2,
            "neck": 3,
            "hat": 4,
            "outer_rsleeve": 5,
            "outer_lsleeve": 6,
            "outer_torso": 7,
            "inner_rsleeve": 8,
            "inner_lsleeve": 9,
            "inner_torso": 10,
            "pants_hip": 11,
            "pants_rsleeve": 12,
            "pants_lsleeve": 13,
            "skirt": 14,
            "right_arm": 15,
            "left_arm": 16,
            "right_shoe": 17,
            "left_shoe": 18,
            "right_leg": 19,
            "left_leg": 20
        }

    def resize_keep_aspect_ratio(self, image, target_size):
        """
        원본 비율을 유지하면서 이미지 리사이징

        Args:
            image: PIL 이미지
            target_size: 목표 크기 (width, height)

        Returns:
            리사이징된 PIL 이미지
        """
        width, height = image.size
        target_width, target_height = target_size

        # 비율 계산
        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 비율 유지하며 리사이징
        resized_img = image.resize((new_width, new_height))

        # 여백을 채운 새 이미지 생성
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        # 중앙에 배치
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))

        return new_img

    def _load_model(self):
        """yisol/IDM-VTON 사전학습 모델 로드 - TryOnAPI 앱 모델 사용"""
        try:
            # 모델 디렉토리 확인
            model_dir = os.path.join(settings.BASE_DIR, "models", "idm_vton")
            idm_dir = os.path.join(model_dir, "yisol_idm_vton")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(idm_dir, exist_ok=True)

            print(f"\n=== IDM-VTON 모델 로드 시작 ===")
            print(f"모델 디렉토리: {model_dir}")

            # 모델 다운로드 여부 확인
            model_files_exist = False
            if os.path.exists(idm_dir):
                print(f"IDM-VTON 모델 디렉토리 확인: {idm_dir}")
                # 필수 모델 파일 확인
                subdirs = ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"]
                found_dirs = [d for d in subdirs if os.path.exists(os.path.join(idm_dir, d))]
                print(f"발견된 모델 디렉토리: {found_dirs}")

                if len(found_dirs) >= len(subdirs):
                    model_files_exist = True
                    print("모든 필수 모델 파일을 찾았습니다.")
                else:
                    print(f"일부 모델 파일이 없습니다. 발견: {len(found_dirs)}/{len(subdirs)}")
            else:
                print(f"IDM-VTON 모델 디렉토리가 없습니다: {idm_dir}")

            # 필요 패키지 로드
            from diffusers import DDPMScheduler, AutoencoderKL
            from transformers import (
                CLIPImageProcessor,
                CLIPVisionModelWithProjection,
                CLIPTextModel,
                CLIPTextModelWithProjection,
                AutoTokenizer
            )

            # 앱 내 모델 파일 임포트
            from app.models.diffusionModels.models.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
            from app.models.diffusionModels.models.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
            from app.models.diffusionModels.models.unet_hacked_tryon import UNet2DConditionModel

            # 모델 ID 또는 로컬 경로 설정
            model_id = os.path.join(settings.BASE_DIR, "models", "idm_vton", "yisol_idm_vton")

            # diffusers 캐시 디렉토리 확인
            import tempfile
            cache_dir = os.path.join(tempfile.gettempdir(), "diffusers_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # TryOnAPI에 맞춰 각 컴포넌트 로드
            print("\n=== 컴포넌트별 모델 로드 시작 ===")

            # CUDA 사용 가능 시 float16, 아닐 경우 float32 사용
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"모델 데이터 타입: {dtype}")

            # UNet 모델 로드
            print("UNet 모델 로드 중...")
            unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=dtype
            )
            unet.requires_grad_(False)

            # UNet 인코더 로드
            print("UNet 인코더 로드 중...")
            unet_encoder = UNet2DConditionModel_ref.from_pretrained(
                model_id,
                subfolder="unet_encoder",
                torch_dtype=dtype
            )
            unet_encoder.requires_grad_(False)

            # 토크나이저 로드
            print("토크나이저 로드 중...")
            tokenizer_one = AutoTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer",
                revision=None,
                use_fast=False
            )
            tokenizer_two = AutoTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer_2",
                revision=None,
                use_fast=False
            )

            # 텍스트 인코더 로드
            print("텍스트 인코더 로드 중...")
            text_encoder_one = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=dtype
            )
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                model_id,
                subfolder="text_encoder_2",
                torch_dtype=dtype
            )
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)

            # 이미지 인코더 로드
            print("이미지 인코더 로드 중...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_id,
                subfolder="image_encoder",
                torch_dtype=dtype
            )
            image_encoder.requires_grad_(False)

            # VAE 로드
            print("VAE 로드 중...")
            vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=dtype
            )
            vae.requires_grad_(False)

            # 스케줄러 로드
            print("스케줄러 로드 중...")
            noise_scheduler = DDPMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler"
            )

            # 파이프라인 구성
            print("파이프라인 구성 중...")
            pipe = TryonPipeline(
                unet=unet,
                unet_encoder=unet_encoder,
                vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
            )

            # UNet 인코더 연결
            pipe.unet_encoder = unet_encoder

            # 디바이스로 이동
            pipe = pipe.to(self.device)

            # 메모리 최적화 옵션 설정
            if torch.cuda.is_available():
                try:
                    # xformers 메모리 효율 활성화 시도
                    pipe.enable_xformers_memory_efficient_attention()
                    print("xformers 메모리 최적화 활성화됨")
                except:
                    # 실패 시 어텐션 슬라이싱 사용
                    pipe.enable_attention_slicing()
                    print("어텐션 슬라이싱 활성화됨")

            print("모델 로드 성공!")
            return pipe

        except Exception as e:
            print(f"모델 로드 프로세스 오류: {str(e)}")
            print(traceback.format_exc())
            raise ValueError(f"모델 로드 실패: {str(e)}")

    def process_images(
        self,
        person_image_path: str,
        clothing_image_path: str,
        parsing_result_path: str,
        densepose_data: Dict[str, Any],
        category: str,
        agnostic_image_path: str = None
    ) -> str:
        """
        가상 착용 처리 파이프라인 - IDM-VTON 모델 사용

        Args:
            person_image_path: 사용자 이미지 경로
            clothing_image_path: 의류 이미지 경로
            parsing_result_path: 인체 파싱 결과 경로
            densepose_data: DensePose 데이터
            category: 의류 카테고리 (upper, lower, dress, outer)
            agnostic_image_path: 의류 영역이 제거된 이미지 경로 (선택사항)

        Returns:
            str: 결과 이미지 경로
        """
        # 장치 준비
        device = self.device
        print(f"사용 장치: {device}")

        print(f"\n=== 가상 착용 처리 시작: {category} 카테고리 ===")
        print(f"사용자 이미지 경로: {person_image_path}")
        print(f"의류 이미지 경로: {clothing_image_path}")
        print(f"파싱 결과 경로: {parsing_result_path}")

        # 결과 디렉토리 준비
        output_dir = os.path.join(settings.OUTPUT_DIR, "idm_vton")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. 이미지 로드
            garm_img_orig = Image.open(clothing_image_path).convert("RGB")
            human_img_orig = Image.open(person_image_path).convert("RGB")

            print(f"사용자 이미지 크기: {human_img_orig.size}")
            print(f"의류 이미지 크기: {garm_img_orig.size}")

            # 원본 크기 저장
            original_size = human_img_orig.size

            # 2. 이미지 크기 조정 - 비율 유지 리사이징
            human_img = self.resize_keep_aspect_ratio(human_img_orig, (768, 1024))
            garm_img = self.resize_keep_aspect_ratio(garm_img_orig, (768, 1024))
            
            # 2-1. Agnostic 이미지 로드 (제공된 경우)
            agnostic_img = None
            if agnostic_image_path and os.path.exists(agnostic_image_path):
                print(f"Agnostic 이미지 로드: {agnostic_image_path}")
                agnostic_img = Image.open(agnostic_image_path).convert("RGB")
                print(f"Agnostic 이미지 크기: {agnostic_img.size}")
                
                # 모델 입력 크기로 리사이징
                if agnostic_img.size != (768, 1024):
                    agnostic_img = self.resize_keep_aspect_ratio(agnostic_img, (768, 1024))
                    print(f"Agnostic 이미지 리사이징: {agnostic_img.size}")

            # 3. 마스크 생성
            try:
                print("파싱 데이터로부터 마스크 생성 중...")
                with open(parsing_result_path, 'r') as f:
                    parsing_data = json.load(f)

                # 파싱 결과에서 마스크 생성
                mask = self._create_mask_from_parsing_result(parsing_data, category)
                print(f"파싱 데이터로부터 마스크 생성 완료: {mask.size}")
            except Exception as e:
                print(f"파싱 마스크 생성 실패: {str(e)}")
                # 실패 시 기본 마스크 생성
                mask = self._create_default_mask((768, 1024), category)
                print("기본 마스크 생성 완료")

            # 마스크 그레이스케일 버전 생성 (시각화용)
            tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
            mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

            # 4. DensePose 처리
            print("DensePose 데이터 처리 중...")

            # DensePose용 이미지 준비 - 비율 유지 리사이징
            human_img_for_dp = self.resize_keep_aspect_ratio(human_img_orig, (384, 512))

            # DensePose 데이터로 이미지 생성
            # DensePose 데이터로 이미지 생성
            if densepose_data and densepose_data.get("processed_visualization"):
                # 직접 생성된 UV 시각화 이미지 사용
                print(f"DensePose UV 시각화 이미지 로드: {densepose_data['processed_visualization']}")
                pose_img = Image.open(densepose_data["processed_visualization"]).convert("RGB")
                # 모델 입력 크기로 리사이징
                pose_img = self.resize_keep_aspect_ratio(pose_img, (768, 1024))
            elif densepose_data and densepose_data.get("uv_coordinates"):
                # 기존 방식 (UV 맵으로 변환)
                print("UV 좌표로 DensePose 이미지 생성")
                uv_map = self._process_densepose_data(densepose_data, category)
                pose_img = self._uv_map_to_image(uv_map)
            else:
                print("DensePose 데이터 없음 - 기본 이미지 생성")
                pose_img = Image.new('RGB', (384, 512), (0, 0, 0))
            # 원본 크기 복원 후 모델 입력 크기로 변환 (비율 유지)
            pose_img = self.resize_keep_aspect_ratio(pose_img, original_size)
            pose_img = self.resize_keep_aspect_ratio(pose_img, (768, 1024))

            # 5. 모델 추론
            print("IDM-VTON 모델 추론 준비 중...")
            pipe = self.model

            # 모델 장치로 이동
            pipe.to(device)
            pipe.unet_encoder.to(device)

            # 텍스트 입력 설정
            garment_des = f"a {category} garment" if category else "clothes"

            with torch.no_grad():
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.inference_mode():
                    # 1. 프롬프트 인코딩
                    prompt = f"model is wearing {garment_des}"
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    # 프롬프트 인코딩
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    # 의류 프롬프트 인코딩
                    prompt = f"a photo of {garment_des}"
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                    # 텐서 변환
                    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device)
                    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device)

                    # 시드 설정
                    seed = 42
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    # 이미지 어댑터 입력 (비율 유지 리사이징)
                    ip_adapter_image = self.resize_keep_aspect_ratio(garm_img_orig, (768, 1024))

                    # 최종 모델 추론
                    print("IDM-VTON 모델 추론 시작...")
                    
                    # Agnostic 이미지가 있는 경우 해당 이미지 사용
                    input_image = agnostic_img if agnostic_img is not None else human_img
                    print(f"모델 입력 이미지 타입: {'Agnostic' if agnostic_img is not None else '원본 사용자 이미지'}")
                    
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device),
                        negative_prompt_embeds=negative_prompt_embeds.to(device),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device),
                        num_inference_steps=30,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_tensor,
                        text_embeds_cloth=prompt_embeds_c.to(device),
                        cloth=garm_tensor,
                        mask_image=mask,
                        image=human_img,  # 수정: agnostic 이미지 또는 원본 이미지 사용
                        height=1024,
                        width=768,
                        ip_adapter_image=ip_adapter_image,
                        guidance_scale=2.0,
                    )[0]

                    # 결과 이미지 가져오기
                    result_img = np.array(images[0])
                    print("IDM-VTON 모델 추론 완료")

            # 6. 결과 저장
            result_id = str(uuid.uuid4())
            result_name = f"{result_id}.jpg"
            result_path = os.path.join(output_dir, result_name)

            print(f"결과 이미지 저장 중: {result_path}")
            try:
                # 결과 이미지 크기와 타입 출력
                print(f"결과 이미지 정보: 크기 {result_img.shape}, 타입 {result_img.dtype}")

                # PIL로 저장
                Image.fromarray(result_img.astype('uint8')).save(result_path)
                print(f"결과 저장 완료: {result_path}")
            except Exception as e:
                print(f"이미지 저장 오류: {str(e)}")
                # OpenCV로 대체 저장 시도
                cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                print(f"OpenCV로 결과 저장 완료: {result_path}")

            return result_path

        except Exception as e:
            print(f"가상 착용 처리 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            raise ValueError(f"가상 착용 처리 오류: {str(e)}")
    
    def apply_category_specific_processing(self, category: str, result_path: str) -> str:
        """
        카테고리별 추가 후처리 적용
        
        Args:
            category: 의류 카테고리 (upper, lower, dress, outer 등)
            result_path: 원본 결과 이미지 경로
            
        Returns:
            str: 후처리된 결과 이미지 경로
        """
        try:
            print(f"카테고리별 후처리 시작: {category}")
            
            # 입력 이미지 확인
            if not os.path.exists(result_path):
                print(f"결과 이미지를 찾을 수 없음: {result_path}")
                return result_path
            
            # 이미지 로드
            result_img = cv2.imread(result_path)
            if result_img is None:
                print(f"결과 이미지 로드 실패: {result_path}")
                return result_path
            
            # 이미지 크기 확인
            height, width = result_img.shape[:2]
            print(f"결과 이미지 크기: {width}x{height}")
            
            # 후처리 작업 (카테고리별 옵션)
            processed_img = result_img.copy()
            
            # 카테고리별 후처리 로직
            if category == "upper":
                # 상의 후처리 (예: 주변 조정)
                pass
            elif category == "lower":
                # 하의 후처리
                pass
            elif category == "dress":
                # 드레스 후처리
                pass
            elif category == "outer":
                # 아우터 후처리
                pass
            
            # 후처리된 이미지 저장
            processed_path = "./result/final.png"  # 동일 경로에 저장 (덮어쓰기)
            cv2.imwrite(processed_path, processed_img)
            
            print(f"카테고리별 후처리 완료: {category}")
            return processed_path
            
        except Exception as e:
            print(f"후처리 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            # 오류 발생 시 원본 경로 반환
            return result_path

    def _uv_map_to_image(self, uv_map):
        """
        UV 맵을 시각화 이미지로 변환

        Args:
            uv_map: UV 좌표 맵 (h, w, 2)

        Returns:
            PIL.Image: 변환된 RGB 이미지
        """
        # UV 맵을 RGB 이미지로 변환 (시각화)
        h, w = uv_map.shape[:2]
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        # 유효한 UV 좌표만 사용
        valid_mask = ~np.all(uv_map == 0, axis=2)

        # RGB 값으로 변환
        if np.any(valid_mask):
            rgb_image[valid_mask, 0] = (uv_map[valid_mask, 0] * 255).astype(np.uint8)  # R = U
            rgb_image[valid_mask, 1] = (uv_map[valid_mask, 1] * 255).astype(np.uint8)  # G = V
            rgb_image[valid_mask, 2] = 128  # B = 중간값

        return Image.fromarray(rgb_image)

    def _get_target_parts_for_category(self, category):
        """
        의류 카테고리에 해당하는 세그멘테이션 부분 리스트 반환
        IDM-VTON 요구사항에 맞게 구현 (상의에는 목 포함, 손목까지 마스킹)

        Args:
            category: 의류 카테고리

        Returns:
            list: 타겟 세그멘테이션 부분 ID 리스트
        """
        if category not in self.category_mapping:
            raise ValueError(f"지원하지 않는 카테고리: {category}")

        # 카테고리에 맞는 세그멘테이션 부분 이름 리스트
        part_names = self.category_mapping[category]

        # 세그멘테이션 부분 이름에 해당하는 클래스 ID 리스트
        target_parts = []
        for name in part_names:
            # 세그멘테이션 클래스에 없는 경우, 매핑
            if name == "upper_body" or name == "upper_clothes":
                # 상의는 목(neck)과 팔 부분(손목까지) 포함 - IDM-VTON 요구사항
                target_parts.extend([self.segmentation_parts[p] for p in [
                    "inner_torso", "inner_rsleeve", "inner_lsleeve", "neck",
                    "right_arm", "left_arm"  # 팔 부분도 포함 (손목까지)
                ]])
            elif name == "lower_body":
                target_parts.extend([self.segmentation_parts[p] for p in [
                    "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt",
                    "right_leg", "left_leg"  # 다리 부분도 포함
                ]])
            elif name == "pants":
                target_parts.extend([self.segmentation_parts[p] for p in [
                    "pants_hip", "pants_rsleeve", "pants_lsleeve",
                    "right_leg", "left_leg"  # 다리 부분 포함
                ]])
            elif name == "coat" or name == "jacket" or name == "outer":
                # 아우터는 안쪽 상의와 소매 위에 덮어써짐
                target_parts.extend([self.segmentation_parts[p] for p in [
                    "outer_torso", "outer_rsleeve", "outer_lsleeve",
                    "inner_torso", "inner_rsleeve", "inner_lsleeve",
                    "neck", "right_arm", "left_arm"  # 목과 팔 부분도 포함
                ]])
            elif name == "dress":
                # 드레스는 상의와 하의를 모두 포함 (목과 팔 포함)
                target_parts.extend([self.segmentation_parts[p] for p in [
                    "inner_torso", "inner_rsleeve", "inner_lsleeve",
                    "neck", "right_arm", "left_arm",  # 상의 부분 (팔 포함)
                    "pants_hip", "skirt", "right_leg", "left_leg"  # 하의 부분
                ]])
            elif name in self.segmentation_parts:
                target_parts.append(self.segmentation_parts[name])

        return list(set(target_parts))  # 중복 제거

    def _create_segmentation_mask(self, parsing_data, target_parts):
        """
        파싱 데이터에서 특정 부분에 대한 세그멘테이션 마스크 생성
        IDM-VTON 요구사항에 맞게 최적화 및 미디어파이프 키포인트 활용

        Args:
            parsing_data: 파싱 결과 데이터
            target_parts: 마스크에 포함할 세그멘테이션 부분 ID 리스트

        Returns:
            numpy.ndarray: 생성된 마스크
        """
        # 이미지 크기 가져오기
        img_size = parsing_data["image_size"]
        width, height = img_size["width"], img_size["height"]

        # 빈 마스크 생성
        mask = np.zeros((height, width), dtype=np.uint8)

        # 클래스 마스크 정보
        class_masks = parsing_data.get("class_masks", {})

        # 클래스 이름 -> ID 매핑 생성
        name_to_id = {name: id for name, id in self.segmentation_parts.items()}

        # 타겟 부분에 해당하는 픽셀 마스킹
        masked_pixels_count = 0
        for part_name, coords in class_masks.items():
            # 세그멘테이션 클래스 ID 찾기
            part_id = name_to_id.get(part_name)

            # 타겟 부분인 경우 마스킹
            if part_id in target_parts and coords:
                for x, y in coords:
                    if 0 <= x < width and 0 <= y < height:
                        mask[y, x] = 255
                        masked_pixels_count += 1

        # 미디어파이프 키포인트가 있는 경우 활용
        keypoints = parsing_data.get("keypoints", None)
        if keypoints:
            print("키포인트 정보를 활용한 마스크 개선")
            mask = self._enhance_mask_with_keypoints(mask, keypoints, target_parts)

        # 마스크가 비어있는 경우, 기본 영역 사용
        if masked_pixels_count < 100:  # 최소 픽셀 수 기준
            print(f"[WARN] 타겟 부분 마스크 픽셀이 부족함 ({masked_pixels_count}개). 기본 영역 사용")

            # 카테고리별 기본 마스크 영역 추정
            if any(p in target_parts for p in [3, 7, 10]):  # neck, outer_torso, inner_torso
                # 상의 영역 추정 (목 포함)
                y_start = int(height * 0.15)  # 목 부분까지 포함
                y_end = int(height * 0.5)
                x_start = int(width * 0.3)
                x_end = int(width * 0.7)
                mask[y_start:y_end, x_start:x_end] = 255

                # 소매 부분 추가
                if any(p in target_parts for p in [5, 6, 8, 9]):  # 소매 부분
                    # 오른쪽 소매
                    y_mid = (y_start + y_end) // 2
                    sleeve_length = int(width * 0.2)
                    mask[y_mid-int(height*0.1):y_mid+int(height*0.1), x_start-sleeve_length:x_start] = 255

                    # 왼쪽 소매
                    mask[y_mid-int(height*0.1):y_mid+int(height*0.1), x_end:x_end+sleeve_length] = 255

            if any(p in target_parts for p in [11, 12, 13, 14, 19, 20]):  # 하의 관련 부분
                # 하의 영역 추정
                y_start = int(height * 0.5)
                y_end = int(height * 0.9)
                x_start = int(width * 0.3)
                x_end = int(width * 0.7)
                mask[y_start:y_end, x_start:x_end] = 255

        # 마스크 개선 (노이즈 제거 및 구멍 채우기)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 경계 부드럽게 처리
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        return mask

    def _enhance_mask_with_keypoints(self, mask, keypoints, target_parts):
        """
        미디어파이프 키포인트를 활용해 마스크 품질 향상
        IDM-VTON-main에서 참조한 로직으로 키포인트 기반 마스크 생성을 구현

        Args:
            mask: 기존 마스크
            keypoints: 키포인트 정보
            target_parts: 목표 부위 ID 리스트

        Returns:
            numpy.ndarray: 개선된 마스크
        """
        h, w = mask.shape[:2]
        enhanced_mask = mask.copy()

        # 상/하의 케이스 구분
        is_upper = any(p in [3, 7, 8, 9, 10] for p in target_parts)  # 목, 상의 관련
        is_lower = any(p in [11, 12, 13, 14, 19, 20] for p in target_parts)  # 하의 관련
        is_dress = is_upper and is_lower  # 드레스는 상의와 하의를 모두 포함
        is_outer = any(p in [5, 6, 7] for p in target_parts)  # 아우터 관련

        # 키포인트가 리스트 형태로 제공된다고 가정
        if isinstance(keypoints, list) and len(keypoints) > 0:
            # MediaPipe 키포인트 매핑
            # 0: nose, 1-10: face, 11: left_shoulder, 12: right_shoulder
            # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
            # 23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee
            # 27: left_ankle, 28: right_ankle

            # 키포인트 좌표 추출 (normalize 형식에서 pixel로 변환)
            pose_data = []
            for kp in keypoints:
                x, y = int(kp.get("x", 0) * w), int(kp.get("y", 0) * h)
                pose_data.append([x, y])
            pose_data = np.array(pose_data)

            # 필요한 키포인트 정의
            idx_map = {
                "nose": 0,
                "left_shoulder": 11, "right_shoulder": 12,
                "left_elbow": 13, "right_elbow": 14,
                "left_wrist": 15, "right_wrist": 16,
                "left_hip": 23, "right_hip": 24,
                "left_knee": 25, "right_knee": 26,
                "left_ankle": 27, "right_ankle": 28
            }

            # PIL 이미지로 변환하여 드로잉 작업
            mask_pil = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask_pil)

            # 상의 처리 - 목 영역 포함, 손목까지 확장
            if is_upper or is_dress or is_outer:
                # 현재 키포인트에서 필요한 좌표 가져오기
                try:
                    shoulder_right = pose_data[idx_map["right_shoulder"]]
                    shoulder_left = pose_data[idx_map["left_shoulder"]]
                    elbow_right = pose_data[idx_map["right_elbow"]]
                    elbow_left = pose_data[idx_map["left_elbow"]]
                    wrist_right = pose_data[idx_map["right_wrist"]]
                    wrist_left = pose_data[idx_map["left_wrist"]]
                    hip_right = pose_data[idx_map["right_hip"]]
                    hip_left = pose_data[idx_map["left_hip"]]

                    # 소매 두께 설정 (이미지 크기에 비례)
                    ARM_LINE_WIDTH = int(60 / 512 * h)  # 기본 설정

                    # 상체 영역 그리기
                    # 몸통 부분
                    body_poly = [
                        tuple(shoulder_left),
                        tuple(shoulder_right),
                        tuple(hip_right),
                        tuple(hip_left)
                    ]
                    draw.polygon(body_poly, fill=255)

                    # 목 영역 포함 (어깨와 코 사이)
                    nose = pose_data[idx_map["nose"]]
                    neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                                int((nose[1] + (shoulder_left[1] + shoulder_right[1]) / 2) / 2))

                    neck_poly = [
                        tuple(shoulder_left),
                        tuple(shoulder_right),
                        neck_top
                    ]
                    draw.polygon(neck_poly, fill=255)

                    # 팔 영역 (손목까지 확장)
                    # 팔을 두껍게 표현하기 위해 라인으로 그림

                    # 왼팔 소매
                    wrist_left_extended = self._extend_arm_mask(wrist_left, elbow_left, 1.2)
                    draw.line(
                        [tuple(wrist_left_extended), tuple(elbow_left), tuple(shoulder_left)],
                        fill=255,
                        width=ARM_LINE_WIDTH,
                        joint="curve"
                    )

                    # 오른팔 소매
                    wrist_right_extended = self._extend_arm_mask(wrist_right, elbow_right, 1.2)
                    draw.line(
                        [tuple(shoulder_right), tuple(elbow_right), tuple(wrist_right_extended)],
                        fill=255,
                        width=ARM_LINE_WIDTH,
                        joint="curve"
                    )

                    # 어깨 원형 추가 (자연스러운 연결을 위해)
                    draw.ellipse(
                        (shoulder_left[0] - ARM_LINE_WIDTH//2, shoulder_left[1] - ARM_LINE_WIDTH//2,
                        shoulder_left[0] + ARM_LINE_WIDTH//2, shoulder_left[1] + ARM_LINE_WIDTH//2),
                        fill=255
                    )
                    draw.ellipse(
                        (shoulder_right[0] - ARM_LINE_WIDTH//2, shoulder_right[1] - ARM_LINE_WIDTH//2,
                        shoulder_right[0] + ARM_LINE_WIDTH//2, shoulder_right[1] + ARM_LINE_WIDTH//2),
                        fill=255
                    )

                except (IndexError, KeyError) as e:
                    print(f"상체 키포인트 처리 중 오류 발생: {str(e)}")

            # 하의 처리 - 바지에서 스커트로 확장 등
            if is_lower or is_dress:
                try:
                    hip_left = pose_data[idx_map["left_hip"]]
                    hip_right = pose_data[idx_map["right_hip"]]
                    knee_left = pose_data[idx_map["left_knee"]]
                    knee_right = pose_data[idx_map["right_knee"]]
                    ankle_left = pose_data[idx_map["left_ankle"]]
                    ankle_right = pose_data[idx_map["right_ankle"]]

                    # 기본 다리 영역 그리기
                    lower_poly = [
                        tuple(hip_left),
                        tuple(hip_right),
                        tuple(knee_right),
                        tuple(knee_left)
                    ]
                    draw.polygon(lower_poly, fill=255)

                    # 무릎 아래 영역 - 스커트로 확장
                    # 무릎과 발목 사이 중점 계산
                    mid_left = (
                        int((knee_left[0] + ankle_left[0]) / 2),
                        int((knee_left[1] + ankle_left[1]) / 2)
                    )
                    mid_right = (
                        int((knee_right[0] + ankle_right[0]) / 2),
                        int((knee_right[1] + ankle_right[1]) / 2)
                    )

                    # 스커트 모양으로 확장 (바깥쪽으로 넓어지는 형태)
                    skirt_width = int(abs(hip_right[0] - hip_left[0]) * 1.2)
                    skirt_left = (mid_left[0] - int(skirt_width/2), mid_left[1])
                    skirt_right = (mid_right[0] + int(skirt_width/2), mid_right[1])

                    skirt_poly = [
                        tuple(knee_left),
                        tuple(knee_right),
                        skirt_right,
                        skirt_left
                    ]
                    draw.polygon(skirt_poly, fill=255)

                    # 바지 다리 부분 (무릎에서 발목)
                    LEG_LINE_WIDTH = int(50 / 512 * h)

                    # 왼쪽 다리
                    draw.line(
                        [tuple(knee_left), tuple(ankle_left)],
                        fill=255,
                        width=LEG_LINE_WIDTH
                    )

                    # 오른쪽 다리
                    draw.line(
                        [tuple(knee_right), tuple(ankle_right)],
                        fill=255,
                        width=LEG_LINE_WIDTH
                    )

                except (IndexError, KeyError) as e:
                    print(f"하체 키포인트 처리 중 오류 발생: {str(e)}")

            # PIL 이미지를 numpy 배열로 변환
            mask_from_keypoints = np.array(mask_pil)

            # 마스크 결합 - 원본 마스크와 키포인트 기반 마스크 통합
            combined_mask = np.logical_or(enhanced_mask, mask_from_keypoints).astype(np.uint8) * 255

            # 마스크 개선 - 노이즈 제거 및 경계 스무딩
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)[1]

            # 결과 반환
            return combined_mask

        # 키포인트가 없는 경우 원본 마스크 반환
        print("키포인트 정보가 없어 원본 마스크 사용")
        return enhanced_mask

    def _extend_arm_mask(self, wrist, elbow, scale=1.2):
        """
        팔 마스크 확장 - 손목 위치를 확장하여 소매를 더 길게 표현

        Args:
            wrist: 손목 위치 좌표 [x, y]
            elbow: 팔꿈치 위치 좌표 [x, y]
            scale: 확장 비율 (기본값 1.2)

        Returns:
            list: 확장된 손목 좌표 [x, y]
        """
        # 벡터 방향으로 확장
        wrist_extended = [
            int(elbow[0] + scale * (wrist[0] - elbow[0])),
            int(elbow[1] + scale * (wrist[1] - elbow[1]))
        ]
        return wrist_extended

    def _create_mask_from_parsing_result(self, parsing_data, category):
        """
        파싱 데이터에서 카테고리에 맞는 마스크 생성

        Args:
            parsing_data: 파싱 결과 데이터
            category: 의류 카테고리

        Returns:
            PIL.Image: 생성된 마스크 이미지
        """
        # 이미지 크기 가져오기
        img_size = parsing_data.get("image_size", {})
        width, height = img_size.get("width", 768), img_size.get("height", 1024)

        # 카테고리에 맞는 세그멘테이션 부분 가져오기
        target_parts = self._get_target_parts_for_category(category)

        # 세그멘테이션 마스크 생성
        mask_np = self._create_segmentation_mask(parsing_data, target_parts)

        # 리사이즈 필요시 적용 (IDM-VTON 입력 크기에 맞게)
        if width != 768 or height != 1024:
            mask_np = cv2.resize(mask_np, (768, 1024), interpolation=cv2.INTER_NEAREST)

        # numpy 배열 -> PIL 이미지 변환
        return Image.fromarray(mask_np)

    def _create_default_mask(self, img_size, category):
        """
        기본 마스크 생성 (파싱 데이터가 없는 경우)
        카테고리별로 적절한 영역에 기본 마스크 생성

        Args:
            img_size: 이미지 크기 (width, height)
            category: 의류 카테고리

        Returns:
            PIL.Image: 생성된 마스크 이미지
        """
        width, height = img_size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # 이미지 중심점
        x_middle = width // 2

        # 카테고리별 기본 영역 정의
        if category == "upper" or category == "outer":
            # 상의/아우터 - 상체 영역 (목 포함)
            y_start = int(height * 0.15)  # 목 부분부터
            y_end = int(height * 0.5)     # 허리 부분까지
            x_start = int(width * 0.2)    # 좌우 여유
            x_end = int(width * 0.8)

            # 몸통 영역
            draw.rectangle([(x_start, y_start), (x_end, y_end)], fill=255)

            # 소매 영역 추가
            sleeve_width = int(width * 0.15)
            sleeve_height = int(height * 0.2)
            mid_y = (y_start + y_end) // 2

            # 왼쪽 소매
            draw.rectangle([
                (x_start - sleeve_width, mid_y - sleeve_height//2),
                (x_start, mid_y + sleeve_height//2)
            ], fill=255)

            # 오른쪽 소매
            draw.rectangle([
                (x_end, mid_y - sleeve_height//2),
                (x_end + sleeve_width, mid_y + sleeve_height//2)
            ], fill=255)

        elif category == "lower":
            # 하의 - 하체 영역
            y_start = int(height * 0.5)   # 허리 부분부터
            y_end = int(height * 0.9)     # 발목 부분까지
            leg_width = int(width * 0.15)
            hip_width = int(width * 0.25)

            # 허리/엉덩이 부분 (위쪽 넓게)
            draw.rectangle([
                (x_middle - hip_width, y_start),
                (x_middle + hip_width, y_start + int(height * 0.1))
            ], fill=255)

            # 다리 부분 (양쪽으로 갈라짐)
            draw.rectangle([
                (x_middle - hip_width, y_start + int(height * 0.1)),
                (x_middle - hip_width + leg_width, y_end)
            ], fill=255)

            draw.rectangle([
                (x_middle + hip_width - leg_width, y_start + int(height * 0.1)),
                (x_middle + hip_width, y_end)
            ], fill=255)

        elif category == "dress":
            # 드레스 - 상체와 하체 모두 포함
            y_start = int(height * 0.15)  # 목 부분부터
            y_end = int(height * 0.85)    # 발목 위까지
            x_start = int(width * 0.25)   # 상체는 좁게
            x_end = int(width * 0.75)

            # 상체 영역
            upper_height = int(height * 0.2)
            draw.rectangle([(x_start, y_start), (x_end, y_start + upper_height)], fill=255)

            # 소매 영역
            sleeve_width = int(width * 0.15)
            sleeve_height = int(height * 0.1)
            sleeve_y = y_start + upper_height // 2

            # 왼쪽 소매
            draw.rectangle([
                (x_start - sleeve_width, sleeve_y - sleeve_height//2),
                (x_start, sleeve_y + sleeve_height//2)
            ], fill=255)

            # 오른쪽 소매
            draw.rectangle([
                (x_end, sleeve_y - sleeve_height//2),
                (x_end + sleeve_width, sleeve_y + sleeve_height//2)
            ], fill=255)

            # 하체 영역 (드레스는 아래로 갈수록 퍼짐)
            skirt_top = y_start + upper_height
            skirt_top_width = int(width * 0.3)
            skirt_bottom_width = int(width * 0.45)

            # 사다리꼴 모양으로 그리기
            draw.polygon([
                (x_middle - skirt_top_width, skirt_top),
                (x_middle + skirt_top_width, skirt_top),
                (x_middle + skirt_bottom_width, y_end),
                (x_middle - skirt_bottom_width, y_end)
            ], fill=255)

        # 마스크 개선
        mask_np = np.array(mask)

        # 마스크 부드럽게 처리
        kernel = np.ones((7, 7), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]

        return Image.fromarray(mask_np)

    def _process_densepose_data(self, densepose_data, category):
        """
        DensePose 데이터를 처리하여 UV 맵 생성

        Args:
            densepose_data: DensePose API 응답 데이터
            category: 의류 카테고리

        Returns:
            numpy.ndarray: UV 맵
        """
        # 이미지 크기
        img_width = densepose_data.get("image_width", self.img_size[0])
        img_height = densepose_data.get("image_height", self.img_size[1])

        # 캔버스 생성 (U, V 좌표를 저장할 맵)
        uv_map = np.zeros((img_height, img_width, 2), dtype=np.float32)

        # 몸통 부분에 해당하는 DensePose 파트 ID
        torso_parts = [1, 2]  # 정면, 후면 몸통
        if category == "upper" or category == "outer":
            torso_parts = [1, 2]  # 몸통
        elif category == "lower":
            torso_parts = [3, 4, 5, 6]  # 다리
        elif category == "dress":
            torso_parts = [1, 2, 3, 4, 5, 6]  # 몸통 + 다리

        # UV 좌표 가져오기
        uv_coords = densepose_data.get("uv_coordinates", [])

        for coord in uv_coords:
            part_id = coord.get("part_id")
            if part_id in torso_parts:
                x, y = coord.get("x"), coord.get("y")
                u, v = coord.get("u"), coord.get("v")

                if 0 <= x < img_width and 0 <= y < img_height:
                    uv_map[y, x] = [u, v]

        # UV 맵 크기 조정 (모델 입력 크기에 맞게)
        uv_map_resized = cv2.resize(uv_map, (384, 512), interpolation=cv2.INTER_LINEAR)

        # 빈 영역(0,0) 채우기 (주변 값으로 보간)
        mask = np.all(uv_map_resized == 0, axis=2)
        for _ in range(5):  # 몇 번의 반복으로 채우기
            for i in range(1, uv_map_resized.shape[0]-1):
                for j in range(1, uv_map_resized.shape[1]-1):
                    if mask[i, j]:
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if not mask[ni, nj]:
                                    neighbors.append(uv_map_resized[ni, nj])
                        if neighbors:
                            uv_map_resized[i, j] = np.mean(neighbors, axis=0)
                            mask[i, j] = False

        return uv_map_resized

# 싱글톤 인스턴스
_idm_vton_service = None

def get_idm_vton_service():
    """IDM VTON 서비스 싱글톤 인스턴스 반환"""
    global _idm_vton_service
    if _idm_vton_service is None:
        _idm_vton_service = IDMVTONService()
    return _idm_vton_service