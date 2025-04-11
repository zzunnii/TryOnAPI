"""
다양한 캔버스 크기로 인체 파싱을 테스트하는 스크립트
"""

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 인체 파싱 관련 모듈
from app.services.human_parsing.processor import get_human_parsing_processor
from app.core.config import get_settings

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='Human Parsing Canvas Size Test')
    parser.add_argument('--image', type=str, default=r"C:\Users\tjdwn\OneDrive\Desktop\tester.jpg", help='입력 이미지 경로')
    parser.add_argument('--output-dir', type=str, default='output/canvas_test', help='출력 디렉토리')
    parser.add_argument('--canvas-sizes', type=str, default="320x640",
                       help='테스트할 캔버스 크기 목록 (콤마로 구분, "None"은 원본 크기)')
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 이미지 로드
    try:
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {args.image}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print(f"[INFO] 원본 이미지 크기: {w}x{h}")
    except Exception as e:
        print(f"[ERROR] 이미지 로드 중 오류 발생: {e}")
        return

    # 인체 파싱 프로세서 가져오기
    processor = get_human_parsing_processor()

    # 캔버스 크기 파싱
    canvas_size_strs = args.canvas_sizes.split(',')
    canvas_sizes = []
    for size_str in canvas_size_strs:
        if size_str.strip() == "None":
            canvas_sizes.append(None)
        else:
            width, height = map(int, size_str.split('x'))
            canvas_sizes.append((width, height))

    # 각 캔버스 크기별로 처리
    results = []
    for i, canvas_size in enumerate(canvas_sizes):
        print(f"[INFO] 캔버스 크기 {i+1}/{len(canvas_sizes)}: {canvas_size if canvas_size else '원본 크기'}")

        # 이미지 처리 시작 시간
        start_time = datetime.now()

        # 인체 파싱 처리
        processed_image, pred_mask, orig_mask, overlay_original = processor.process_and_segment(
            image_rgb,
            canvas_size=canvas_size
        )

        # 처리 종료 시간
        end_time = datetime.now()
        process_time = (end_time - start_time).total_seconds()
        print(f"[INFO] 처리 시간: {process_time:.2f}초")

        # 결과 저장
        if overlay_original is not None:
            size_str = f"{canvas_size[0]}x{canvas_size[1]}" if canvas_size else "original"

            # 오버레이 이미지 저장
            overlay_path = os.path.join(args.output_dir, f"canvas_{size_str}.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))

            # 마스크 이미지 저장
            mask_path = os.path.join(args.output_dir, f"mask_{size_str}.png")
            cv2.imwrite(mask_path, orig_mask)

            # JSON 마스크 생성 및 저장
            mask_json = processor.mask_to_json(orig_mask, (w, h))
            json_path = os.path.join(args.output_dir, f"mask_{size_str}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(mask_json, f, ensure_ascii=False, indent=2)

            print(f"[INFO] JSON 마스크 저장 완료: {json_path}")

            # 클래스 수 계산
            class_count = len([c for c in np.unique(orig_mask) if c > 0])
            print(f"[INFO] 감지된 클래스 수: {class_count}")

            results.append({
                'canvas_size': canvas_size,
                'display_name': size_str,
                'process_time': process_time,
                'overlay_path': overlay_path,
                'overlay_image': overlay_original,
                'mask_path': mask_path,
                'json_path': json_path,
                'class_count': class_count
            })

    # 모든 결과 비교 이미지 생성
    if results:
        plt.figure(figsize=(15, 5 * len(results)))

        # 원본 이미지 표시
        plt.subplot(len(results) + 1, 2, 1)
        plt.title('원본 이미지')
        plt.imshow(image_rgb)
        plt.axis('off')

        # 각 캔버스 크기별 결과 표시
        for i, result in enumerate(results):
            plt.subplot(len(results) + 1, 2, i*2 + 3)
            plt.title(f"캔버스 크기: {result['display_name']} (클래스: {result['class_count']}개)")
            plt.imshow(result['overlay_image'])
            plt.axis('off')

        # 저장
        comparison_path = os.path.join(args.output_dir, "all_canvas_comparison.png")
        plt.tight_layout()
        plt.savefig(comparison_path)
        print(f"[INFO] 비교 이미지 저장 완료: {comparison_path}")

    print("[INFO] 테스트 완료!")

if __name__ == '__main__':
    main()