"""
인체 파싱 프로세서 - BiRefNet + 인체 파싱 모델 통합
"""

import os
import cv2
import torch
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ...models.parsingModels.parsing_model import ParsingModel
from .birefnet import setup_model, load_dataset_stats, process_image_for_segmentation

from app.core.config import get_settings

settings = get_settings()

class HumanParsingProcessor:
    def __init__(
            self,
            checkpoint_path=None,
            num_classes=21,
            device=None,
            input_size=None,
            birefnet_token=None,
            stats_file=None
    ):
        """인체 파싱 프로세서 초기화"""
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size or (320, 640)  # 기본값 설정, 실제로는 동적으로 사용됨

        # 체크포인트 경로가 없으면 기본 경로 사용
        if checkpoint_path is None:
            checkpoint_path = os.path.join(settings.SEGMENTATION_MODEL_PATH, 'human_parsing_model.pth')

        # 파싱 모델 로드
        self.model = ParsingModel(
            num_classes=num_classes,
            backbone_pretrained=False,
            fpn_channels=256,
            decoder_channels=512
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # BiRefNet 모델 로드
        self.birefnet_model = setup_model(token=birefnet_token)

        # 통계 파일이 없으면 기본 경로 사용
        if stats_file is None:
            stats_file = os.path.join(settings.BASE_DIR, "utils", "statistics_summary.json")

        self.dataset_stats = load_dataset_stats(stats_file)

        # 전처리 변환 정의 (동적으로 적용됨)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # 클래스 정의
        self.class_names = [
            "background", "hair", "face", "neck", "hat",
            "outer_rsleeve", "outer_lsleeve", "outer_torso",
            "inner_rsleeve", "inner_lsleeve", "inner_torso",
            "pants_hip", "pants_rsleeve", "pants_lsleeve",
            "skirt", "right_arm", "left_arm",
            "right_shoe", "left_shoe", "right_leg", "left_leg"
        ]

    def get_transform(self, input_size=None):
        """이미지 크기에 따른 변환 정의"""
        if input_size is None:
            input_size = self.input_size

        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    @torch.no_grad()
    def predict(self, image):
        """이미지를 입력받아 세그멘테이션 예측"""
        # RGBA 이미지를 RGB로 변환
        if image.shape[2] == 4:
            rgb_image = image[:, :, :3].copy()
        else:
            rgb_image = image.copy()

        # 원본 크기 저장
        original_h, original_w = rgb_image.shape[:2]

        # 이미지 전처리
        transform = self.get_transform()
        transformed = transform(image=rgb_image)
        x = transformed['image'].unsqueeze(0).to(self.device)

        # 예측 수행
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        return pred, rgb_image

    def remove_small_regions(self, mask, min_area=20):
        """작은 영역 제거"""
        cleaned_mask = np.zeros_like(mask)
        for class_id in range(1, self.num_classes):
            # 각 클래스별 마스크
            class_mask = (mask == class_id).astype(np.uint8)

            # 너무 작은 영역 제거
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            # 연결 컴포넌트 분석
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)

            # 충분히 큰 영역만 유지
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    cleaned_mask[labels == i] = class_id

        return cleaned_mask

    def smooth_boundaries(self, mask):
        """클래스 간 경계 개선"""
        # 경계 검출
        edges = np.zeros_like(mask, dtype=np.uint8)
        for class_id in range(1, mask.max() + 1):
            class_mask = (mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 1, thickness=2)

        # 경계 영역 확장
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        blending_region = dilated_edges > 0

        # 경계 개선
        improved_mask = mask.copy()
        y_indices, x_indices = np.where(blending_region)
        for y, x in zip(y_indices, x_indices):
            neighborhood = mask[max(0, y - 1):min(mask.shape[0], y + 2),
                           max(0, x - 1):min(mask.shape[1], x + 2)]
            if neighborhood.size > 0:
                values, counts = np.unique(neighborhood, return_counts=True)
                if len(values) > 1 or (len(values) == 1 and values[0] != 0):
                    most_common = values[np.argmax(counts)]
                    improved_mask[y, x] = most_common

        return improved_mask

    def apply_edge_processing(self, image, mask, kernel_size=5, sigma=1.0):
        """마스크 경계 부분을 부드럽게 처리"""
        if len(mask.shape) == 3:
            mask_gray = mask[:, :, 0]
        else:
            mask_gray = mask

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_gray, kernel, iterations=1)
        eroded = cv2.erode(mask_gray, kernel, iterations=1)
        edges = dilated - eroded

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        edges_3d = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
        weight_map = edges_3d.astype(np.float32) / 255.0

        result = image.copy()
        for i in range(3):
            result[:, :, i] = (
                image[:, :, i] * (1 - weight_map[:, :, i]) +
                blurred[:, :, i] * weight_map[:, :, i]
            )
        return result

    def visualize_segmentation(self, image, mask, alpha=0.7):
        """세그멘테이션 결과 시각화"""
        overlay = np.zeros_like(image)
        for class_id in range(1, self.num_classes):
            if class_id in np.unique(mask):
                # 클래스별 색상 (일관성 있는 색상 사용)
                color = ((class_id * 37) % 256, (class_id * 73) % 256, (class_id * 113) % 256)
                overlay[mask == class_id] = color

        # 블렌딩
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return output

    def mask_to_json(self, mask, original_size):
        """세그멘테이션 마스크를 JSON으로 변환"""
        # 클래스별 픽셀 좌표 추출
        class_masks = {}
        for class_id in range(1, self.num_classes):
            if class_id in np.unique(mask):
                # 클래스 이름
                class_name = self.class_names[class_id]

                # 해당 클래스 픽셀 좌표 추출
                y_indices, x_indices = np.where(mask == class_id)

                # 좌표 목록 생성
                coords = []
                for x, y in zip(x_indices, y_indices):
                    coords.append([int(x), int(y)])

                # 클래스별 마스크 정보 저장
                if coords:
                    class_masks[class_name] = coords

        # 결과 JSON
        result = {
            'image_size': {'width': int(original_size[0]), 'height': int(original_size[1])},
            'class_masks': class_masks
        }

        return result

    def process_and_segment(self, image_rgb, canvas_size=None):
        """배경 제거 -> 세그멘테이션 -> 원본 좌표 변환 파이프라인"""
        # 이미지 로드
        if isinstance(image_rgb, str):
            original_image = cv2.imread(image_rgb)
            if original_image is None:
                raise ValueError(f"이미지 로드 실패: {image_rgb}")
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_rgb  # numpy 배열로 입력된 경우

        orig_h, orig_w = original_image.shape[:2]
        print(f"[INFO] Original image dimensions: {orig_w}x{orig_h}")

        # canvas_size가 None이면 원본 이미지 크기 사용
        if canvas_size is None:
            canvas_size = (orig_w, orig_h)  # (width, height)

        # canvas_size 형식 확인 (항상 width, height 순서로 사용)
        print(f"[INFO] Using canvas size: {canvas_size[0]}x{canvas_size[1]}")

        # 배경 제거 및 리사이징
        processed_image, person_mask, original_img, original_mask, transform_info = process_image_for_segmentation(
            self.birefnet_model,
            original_image,
            self.dataset_stats,
            final_canvas=canvas_size
        )

        if transform_info is None:
            # 사람/물체가 감지되지 않은 경우
            print("[WARN] 이미지에서 객체를 찾을 수 없습니다.")
            return None, None, None, None

        # 세그멘테이션 예측
        pred_mask, _ = self.predict(processed_image)

        # 후처리
        clean_pred_mask = self.remove_small_regions(pred_mask, min_area=20)
        smoothed_mask = self.smooth_boundaries(clean_pred_mask)

        # 안티앨리어싱 및 결과 생성
        antialiased_image = self.apply_edge_processing(processed_image, person_mask, kernel_size=5, sigma=1.5)

        # 시각화를 위한 오버레이 생성 (처리된 이미지에 대한)
        overlay_processed = self.visualize_segmentation(
            image=antialiased_image,
            mask=smoothed_mask,
            alpha=0.7
        )

        # === 여기서부터 마스크를 원본 좌표계로 변환하는 로직 추가 ===

        # 캔버스 위치 정보
        x, y, pw, ph = transform_info['canvas_placement']

        # 캔버스에서 객체 영역 추출
        canvas_mask = np.zeros_like(smoothed_mask)
        canvas_mask[y:y + ph, x:x + pw] = smoothed_mask[y:y + ph, x:x + pw]

        # 원본 이미지의 바운딩 박스와 크기
        orig_x, orig_y, orig_x2, orig_y2 = transform_info['original_bbox']
        orig_w, orig_h = transform_info['original_size']

        # 원본 크기에 맞는 마스크 생성
        orig_mask = np.zeros((orig_h, orig_w), dtype=smoothed_mask.dtype)

        # 원본 이미지의 객체 영역에 맞춰 마스크 리사이징 및 위치 조정
        obj_w, obj_h = orig_x2 - orig_x + 1, orig_y2 - orig_y + 1
        if pw > 0 and ph > 0:
            # 마스크를 캔버스 위치에서 추출해 원본 객체 크기로 리사이징
            obj_mask = cv2.resize(canvas_mask[y:y + ph, x:x + pw], (obj_w, obj_h), interpolation=cv2.INTER_NEAREST)
            # 원본 위치에 배치
            orig_mask[orig_y:orig_y2 + 1, orig_x:orig_x2 + 1] = obj_mask

        # 원본 이미지에 마스크 적용한 시각화 결과
        overlay_original = self.visualize_segmentation(
            image=original_img,
            mask=orig_mask,
            alpha=0.7
        )

        return processed_image, smoothed_mask, orig_mask, overlay_original

    def process_image(self, image_path, output_dir=None, save_json=True, canvas_size=None):
        """전체 처리 파이프라인: 배경 제거 -> 세그멘테이션 -> JSON 반환"""
        # 이미지 로드
        if isinstance(image_path, str):
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path  # numpy 배열로 입력된 경우
            file_name = "processed_image"

        orig_h, orig_w = original_image.shape[:2]

        # 세그멘테이션 프로세스 실행
        processed_image, processed_mask, orig_mask, overlay_original = self.process_and_segment(
            original_image,
            canvas_size=canvas_size  # 지정된 캔버스 크기 사용
        )

        if processed_image is None:
            # 실패한 경우 빈 결과 반환
            return {
                'image_size': {'width': orig_w, 'height': orig_h},
                'class_masks': {}
            }, original_image, np.zeros((orig_h, orig_w), dtype=np.uint8)

        # JSON 변환 (원본 크기의 마스크 사용)
        result_json = self.mask_to_json(orig_mask, (orig_w, orig_h))

        # 결과 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # JSON 저장
            if save_json:
                json_path = os.path.join(output_dir, f"{file_name}_mask.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=2)
                print(f"JSON 마스크 저장 완료: {json_path}")

            # 시각화 이미지 저장 (원본 기반 오버레이)
            vis_path = os.path.join(output_dir, f"{file_name}_visualization.png")
            cv2.imwrite(vis_path, cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))

            # 비교 이미지 저장 (원본 + 오버레이)
            comparison = np.hstack((
                cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR)
            ))
            comp_path = os.path.join(output_dir, f"{file_name}_comparison.png")
            cv2.imwrite(comp_path, comparison)

        return result_json, processed_image, orig_mask

# 싱글톤 인스턴스 생성
_processor_instance = None

def get_human_parsing_processor():
    """싱글톤 인스턴스 반환"""
    global _processor_instance
    if _processor_instance is None:
        # 설정에서 필요한 경로 가져오기
        settings_obj = get_settings()
        checkpoint_path = os.path.join(settings_obj.SEGMENTATION_MODEL_PATH, 'human_parsing_model.pth')
        stats_file = os.path.join(settings_obj.BASE_DIR, 'utils', 'statistics_summary.json')
        
        # 프로세서 인스턴스 생성
        _processor_instance = HumanParsingProcessor(
            checkpoint_path=checkpoint_path,
            stats_file=stats_file
        )
    
    return _processor_instance