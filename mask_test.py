import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw


def create_agnostic_image_long_dress(
        image_path,
        parsing_json_path,
        keypoints_json_path,
        output_path=None
):
    """
    롱 드레스 카테고리에 맞게 agnostic 이미지 생성
    (의류 영역이 1(검은색), 배경이 0(원본)인 마스크 적용)

    Args:
        image_path: 원본 이미지 경로
        parsing_json_path: 인체 파싱 결과 JSON 파일 경로
        keypoints_json_path: 미디어파이프 키포인트 JSON 파일 경로
        output_path: 출력 이미지 경로 (기본값: None, 지정하지 않으면 반환만 함)

    Returns:
        PIL.Image: 생성된 agnostic 이미지
    """
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    # RGB로 변환 (OpenCV는 BGR 포맷을 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # 2. 파싱 결과 로드
    with open(parsing_json_path, 'r') as f:
        parsing_data = json.load(f)

    # 3. 키포인트 로드
    with open(keypoints_json_path, 'r') as f:
        keypoints_data = json.load(f)

    keypoints = keypoints_data.get("pose_landmarks", [])

    print(f"이미지 크기: {width}x{height}")
    print(f"파싱 데이터 키: {list(parsing_data.keys())}")
    print(f"키포인트 수: {len(keypoints)}")

    # 4. 의류 영역 마스크 생성 (의류:1, 배경:0)
    # 롱 드레스의 경우 상체 + 하체 영역 (보존 부위 제외)
    clothing_categories = [
        "outer_rsleeve", "outer_lsleeve", "outer_torso", "neck",
        "inner_rsleeve", "inner_lsleeve", "inner_torso",
        "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt"
    ]

    # 보존할 부위 카테고리
    preserve_categories = ["face", "hair", "right_arm", "left_arm", "right_leg", "left_leg"]

    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }

    # 의류 영역(1)과 배경(0) 마스크 생성
    mask = np.zeros((height, width), dtype=np.uint8)  # 초기값 0 (배경)

    # 파싱 결과에서 의류 영역을 1로 설정
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 1  # 의류 영역은 1로 설정

    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업 (0-255 값 범위로 변환)
    mask_pil = Image.fromarray(mask * 255)
    draw = ImageDraw.Draw(mask_pil)

    # 키포인트 좌표 추출 (normalize 형식에서 pixel로 변환)
    pose_data = []
    for kp in keypoints:
        x, y = int(kp.get("x", 0) * width), int(kp.get("y", 0) * height)
        pose_data.append([x, y])

    pose_data = np.array(pose_data)

    # 필요한 키포인트 인덱스 매핑
    idx_map = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28
    }

    try:
        # 상체 영역 확장
        shoulder_right = pose_data[idx_map["right_shoulder"]]
        shoulder_left = pose_data[idx_map["left_shoulder"]]
        elbow_right = pose_data[idx_map["right_elbow"]]
        elbow_left = pose_data[idx_map["left_elbow"]]
        wrist_right = pose_data[idx_map["right_wrist"]]
        wrist_left = pose_data[idx_map["left_wrist"]]
        hip_right = pose_data[idx_map["right_hip"]]
        hip_left = pose_data[idx_map["left_hip"]]

        # 소매 두께 설정 (이미지 크기에 비례)
        ARM_LINE_WIDTH = int(80 / 512 * height)  # 더 두껍게 설정

        # 상체 영역 그리기 (몸통 부분)
        body_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            tuple(hip_right),
            tuple(hip_left)
        ]
        draw.polygon(body_poly, fill=255)  # 1(255)로 마스킹

        # 목 영역 (어깨와 코 사이)
        nose = pose_data[idx_map["nose"]]
        neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                    int((nose[1] * 0.8 + (shoulder_left[1] + shoulder_right[1]) / 2 * 0.2)))

        # 목 주변 더 넓게 마스킹 (이미지 2처럼)
        neck_width = int(abs(shoulder_right[0] - shoulder_left[0]) * 0.5)
        neck_poly = [
            (shoulder_left[0] - neck_width // 4, shoulder_left[1]),
            (shoulder_right[0] + neck_width // 4, shoulder_right[1]),
            (neck_top[0] + neck_width // 2, neck_top[1] - neck_width // 3),
            (neck_top[0] - neck_width // 2, neck_top[1] - neck_width // 3)
        ]
        draw.polygon(neck_poly, fill=255)  # 1(255)로 마스킹

        # 팔 영역 확장 (손목까지)
        # 왼팔 소매 - 자연스럽게 넓게
        sleeve_left_points = [
            (shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1]),
            (elbow_left[0] - ARM_LINE_WIDTH // 2, elbow_left[1]),
            (wrist_left[0] - ARM_LINE_WIDTH // 3, wrist_left[1] + ARM_LINE_WIDTH // 3),
            (wrist_left[0] + ARM_LINE_WIDTH // 2, wrist_left[1] + ARM_LINE_WIDTH // 3),
            (elbow_left[0] + ARM_LINE_WIDTH // 2, elbow_left[1]),
            (shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1])
        ]
        draw.polygon(sleeve_left_points, fill=255)

        # 오른팔 소매 - 자연스럽게 넓게
        sleeve_right_points = [
            (shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1]),
            (elbow_right[0] - ARM_LINE_WIDTH // 2, elbow_right[1]),
            (wrist_right[0] - ARM_LINE_WIDTH // 2, wrist_right[1] + ARM_LINE_WIDTH // 3),
            (wrist_right[0] + ARM_LINE_WIDTH // 3, wrist_right[1] + ARM_LINE_WIDTH // 3),
            (elbow_right[0] + ARM_LINE_WIDTH // 2, elbow_right[1]),
            (shoulder_right[0] + ARM_LINE_WIDTH // 2, shoulder_right[1])
        ]
        draw.polygon(sleeve_right_points, fill=255)

        # 하체 영역 확장
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        ankle_left = pose_data[idx_map["left_ankle"]]
        ankle_right = pose_data[idx_map["right_ankle"]]

        # 하체 영역 그리기
        lower_width = int(abs(hip_right[0] - hip_left[0]) * 1.2)  # 더 넓게
        lower_left = (hip_left[0] - lower_width // 5, hip_left[1])
        lower_right = (hip_right[0] + lower_width // 5, hip_right[1])

        # 무릎 위치도 더 넓게
        knee_left_wide = (knee_left[0] - lower_width // 5, knee_left[1])
        knee_right_wide = (knee_right[0] + lower_width // 5, knee_right[1])

        # 하체 폴리곤 (허리-무릎)
        lower_poly = [
            lower_left,
            lower_right,
            knee_right_wide,
            knee_left_wide
        ]
        draw.polygon(lower_poly, fill=255)

        # 롱 드레스 - 발목까지 확장 (이미지 2와 유사하게)
        # 발목 위치 - 더 넓게
        LEG_LINE_WIDTH = int(80 / 512 * height)
        ankle_margin = int(LEG_LINE_WIDTH * 1.5)

        # 스커트/드레스 하단부 (무릎-발목)
        skirt_bottom = ankle_left[1] - LEG_LINE_WIDTH // 2  # 발목 위에서 끝나도록
        skirt_left = (ankle_left[0] - ankle_margin, skirt_bottom)
        skirt_right = (ankle_right[0] + ankle_margin, skirt_bottom)

        # 스커트 폴리곤 (무릎-발목)
        skirt_poly = [
            knee_left_wide,
            knee_right_wide,
            skirt_right,
            skirt_left
        ]
        draw.polygon(skirt_poly, fill=255)

        # 발목 위쪽에 자연스러운 경계선 추가
        transition_height = LEG_LINE_WIDTH
        transition_bottom = skirt_bottom + transition_height

        # 더 자연스러운 전환을 위한 그라데이션 효과
        steps = 10
        for i in range(steps):
            alpha = 255 * (steps - i) / steps  # 점차 투명해지는 효과
            y_pos = skirt_bottom + i * transition_height / steps
            transition_width = ankle_margin * (steps - i) / steps

            left_x = ankle_left[0] - transition_width
            right_x = ankle_right[0] + transition_width

            draw.line([(left_x, y_pos), (right_x, y_pos)], fill=int(alpha), width=2)

    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")

    # PIL 이미지를 numpy 배열로 변환하고 0-1 범위로 정규화
    mask_enhanced = np.array(mask_pil) / 255.0

    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    mask_enhanced = mask_enhanced.astype(np.float32)

    # 가우시안 블러로 경계 부드럽게
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (15, 15), 0)

    # 이진화 (임계값 0.3 - 더 넓게 마스킹)
    mask_enhanced = (mask_enhanced > 0.3).astype(np.uint8)

    # 모폴로지 연산 - 마스크 확장
    kernel = np.ones((7, 7), np.uint8)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 더 확장
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)

    # 다시 부드럽게
    mask_enhanced = cv2.GaussianBlur(mask_enhanced.astype(np.float32), (11, 11), 0)
    mask_enhanced = (mask_enhanced > 0.3).astype(np.uint8)

    # 6. Agnostic 이미지 생성 (의류 영역을 검은색으로 마스킹)
    # 마스크가 1인 영역(의류 영역)을 검은색으로 설정
    agnostic_image = image_rgb.copy()
    agnostic_image[mask_enhanced == 1] = [0, 0, 0]

    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)

    # 디버깅: 마스크 이미지 저장 (선택 사항)
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, mask_enhanced * 255)
        print(f"마스크 이미지 저장: {mask_path}")

    # 7. 결과 저장
    if output_path:
        agnostic_pil.save(output_path)
        print(f"Agnostic 이미지 저장: {output_path}")

    return agnostic_pil


# 메인 실행 부분
if __name__ == "__main__":
    # 파일 경로 설정
    image_path = r"C:\Users\tjdwn\GitHub\TryOnAPI\temp\user_images\1eefb4ba-5bf0-4e3e-960b-b5c6cab1918e_1008_A003_000.jpg"
    parsing_json_path = r"C:\Users\tjdwn\GitHub\TryOnAPI\output\human_parsing\1e514858-ed81-4efd-9761-134459dac8e0_mask.json"
    keypoints_json_path = r"C:\Users\tjdwn\GitHub\TryOnAPI\output\mediapipe\2eb742ab-d094-4d8b-9fdf-e9c4cbc6699a_keypoints.json"

    # 결과 저장 경로
    output_dir = r"C:\Users\tjdwn\GitHub\TryOnAPI\output\agnostic"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "long_dress_agnostic.jpg")

    # Agnostic 이미지 생성
    agnostic_image = create_agnostic_image_long_dress(
        image_path,
        parsing_json_path,
        keypoints_json_path,
        output_path
    )

    print("Agnostic 이미지 생성 완료!")