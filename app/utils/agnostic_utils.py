"""
Agnostic 이미지 생성 유틸리티
의류 가상착용(Virtual Try-On)을 위한 agnostic 이미지 생성 유틸리티 함수들

agnostic 이미지: 의류 영역이 검은색(0)으로 마스킹된 이미지
마스크: 의류 영역이 1, 배경이 0으로 설정된 이진 마스크
"""
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw
import traceback

def create_agnostic_image(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    category="long_dress",
    output_path=None
):
    """
    카테고리에 맞는 agnostic 이미지(의류 영역만 0으로 마스킹된 이미지) 생성
    
    Args:
        image_path: 원본 이미지 경로
        parsing_json_path: 인체 파싱 결과 JSON 파일 경로
        keypoints_json_path: 미디어파이프 키포인트 JSON 파일 경로
        category: 의류 카테고리 (upper, lower, long_upper, dress, long_dress, skirt, short_skirt, long_skirt)
        output_path: 출력 이미지 경로 (기본값: None, 지정하지 않으면 반환만 함)
    
    Returns:
        tuple: (PIL.Image - 생성된 agnostic 이미지, np.ndarray - 생성된 마스크(의류:1, 배경:0))
    """
    # 카테고리별 함수 매핑
    category_functions = {
        "upper": create_agnostic_upper,
        "long_upper": create_agnostic_long_upper,
        "lower": create_agnostic_lower,
        "dress": create_agnostic_dress,
        "long_dress": create_agnostic_long_dress,
        "skirt": create_agnostic_skirt,
        "short_skirt": create_agnostic_short_skirt,
        "long_skirt": create_agnostic_long_skirt
    }
    
    # 카테고리 확인 및 기본값 설정
    if category not in category_functions:
        print(f"지원되지 않는 카테고리: {category}, 기본값 'upper'로 설정")
        category = "upper"
    
    # 해당 카테고리의 함수 호출
    try:
        return category_functions[category](
            image_path,
            parsing_json_path,
            keypoints_json_path,
            output_path
        )
    except Exception as e:
        print(f"Agnostic 이미지 생성 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        # 오류 발생 시 기본 함수 호출 시도
        return create_agnostic_basic(
            image_path,
            parsing_json_path,
            keypoints_json_path,
            output_path
        )

def create_agnostic_basic(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    기본 agnostic 이미지 생성 (의류 카테고리 미지정 시)
    """
    try:
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
        
        # 3. 의류 영역(제거할 부분) 마스크 생성
        clothing_categories = [
            "outer_rsleeve", "outer_lsleeve", "outer_torso", "neck",
            "inner_rsleeve", "inner_lsleeve", "inner_torso",
            "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt"
        ]
        
        # 카테고리 ID 매핑
        category_ids = {
            "hair": 1, "face": 2, "neck": 3, "hat": 4,
            "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
            "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
            "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
            "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
            "right_leg": 19, "left_leg": 20
        }
        
        # 빈 마스크 생성 (0: 마스킹, 255: 유지)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 파싱 결과에서 의류 영역 마스킹
        class_masks = parsing_data.get("class_masks", {})
        for class_name, coords in class_masks.items():
            if class_name in clothing_categories and coords:
                for x, y in coords:
                    if 0 <= x < width and 0 <= y < height:
                        mask[y, x] = 0
        
        # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # 4. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
        agnostic_image = image_rgb.copy()
        agnostic_image[mask == 0] = [0, 0, 0]
        
        # PIL 이미지로 변환
        agnostic_pil = Image.fromarray(agnostic_image)
        
        # IDM VTON용 마스크 생성 (의류:1, 배경:0)
        idm_mask = (mask == 0).astype(np.uint8)
        
        # 결과 저장
        if output_path:
            agnostic_pil.save(output_path)
            mask_path = output_path.replace('.jpg', '_mask.png')
            cv2.imwrite(mask_path, idm_mask * 255)
            print(f"Agnostic 이미지 저장: {output_path}")
            print(f"마스크 이미지 저장: {mask_path}")
        
        return agnostic_pil, idm_mask
        
    except Exception as e:
        print(f"기본 Agnostic 이미지 생성 중 오류 발생: {str(e)}")
        # 오류 발생 시 검은색 이미지와 빈 마스크 반환
        black_image = Image.new('RGB', (512, 512), (0, 0, 0))
        empty_mask = np.zeros((512, 512), dtype=np.uint8)
        return black_image, empty_mask

def create_agnostic_upper(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    일반 상의 카테고리에 맞는 agnostic 이미지 생성
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
    print(f"키포인트 수: {len(keypoints)}")
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 상의: 상체 영역
    clothing_categories = [
        "outer_rsleeve", "outer_lsleeve", "outer_torso", "neck",
        "inner_rsleeve", "inner_lsleeve", "inner_torso"
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
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
        "left_hip": 23, "right_hip": 24
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
        ARM_LINE_WIDTH = int(60 / 512 * height)
        
        # 상체 영역 그리기 (몸통 부분)
        body_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            tuple(hip_right),
            tuple(hip_left)
        ]
        draw.polygon(body_poly, fill=0)  # 0으로 마스킹
        
        # 목 영역 (어깨와 코 사이)
        nose = pose_data[idx_map["nose"]]
        neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                    int((nose[1] + (shoulder_left[1] + shoulder_right[1]) / 2) / 2))
        
        neck_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            neck_top
        ]
        draw.polygon(neck_poly, fill=0)  # 0으로 마스킹
        
        # 팔 영역 확장 (손목까지)
        # 왼팔 소매
        draw.line(
            [tuple(shoulder_left), tuple(elbow_left), tuple(wrist_left)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
        # 오른팔 소매
        draw.line(
            [tuple(shoulder_right), tuple(elbow_right), tuple(wrist_right)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_long_upper(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    긴 상의 카테고리(힙 영역 포함)에 맞는 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 긴 상의: 상체 + 힙 영역
    clothing_categories = [
        "outer_rsleeve", "outer_lsleeve", "outer_torso", "neck",
        "inner_rsleeve", "inner_lsleeve", "inner_torso",
        "pants_hip"  # 힙 영역 추가
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
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
        "left_hip": 23, "right_hip": 24
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
        ARM_LINE_WIDTH = int(60 / 512 * height)
        
        # 상체 영역 그리기 (몸통 + 힙 부분)
        # 힙 아래까지 확장 (일반 상의보다 길게)
        hip_extension = int(height * 0.05)  # 힙 아래로 5% 확장
        body_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            (hip_right[0], hip_right[1] + hip_extension),
            (hip_left[0], hip_left[1] + hip_extension)
        ]
        draw.polygon(body_poly, fill=0)  # 0으로 마스킹
        
        # 목 영역 (어깨와 코 사이)
        nose = pose_data[idx_map["nose"]]
        neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                    int((nose[1] + (shoulder_left[1] + shoulder_right[1]) / 2) / 2))
        
        neck_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            neck_top
        ]
        draw.polygon(neck_poly, fill=0)  # 0으로 마스킹
        
        # 팔 영역 확장 (손목까지)
        # 왼팔 소매
        draw.line(
            [tuple(shoulder_left), tuple(elbow_left), tuple(wrist_left)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
        # 오른팔 소매
        draw.line(
            [tuple(shoulder_right), tuple(elbow_right), tuple(wrist_right)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_lower(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    하의 카테고리에 맞는 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 하의: 하체 영역
    clothing_categories = [
        "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt"
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_pil)
    
    # 키포인트 좌표 추출 (normalize 형식에서 pixel로 변환)
    pose_data = []
    for kp in keypoints:
        x, y = int(kp.get("x", 0) * width), int(kp.get("y", 0) * height)
        pose_data.append([x, y])
    
    pose_data = np.array(pose_data)
    
    # 필요한 키포인트 인덱스 매핑
    idx_map = {
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28
    }
    
    try:
        # 하체 영역 확장
        hip_right = pose_data[idx_map["right_hip"]]
        hip_left = pose_data[idx_map["left_hip"]]
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        ankle_left = pose_data[idx_map["left_ankle"]]
        ankle_right = pose_data[idx_map["right_ankle"]]
        
        # 하체 영역 그리기
        LEG_LINE_WIDTH = int(50 / 512 * height)
        
        # 엉덩이부터 무릎까지
        lower_poly = [
            tuple(hip_left),
            tuple(hip_right),
            tuple(knee_right),
            tuple(knee_left)
        ]
        draw.polygon(lower_poly, fill=0)  # 0으로 마스킹
        
        # 다리 부분 (무릎에서 발목)
        # 왼쪽 다리
        draw.line(
            [tuple(knee_left), tuple(ankle_left)],
            fill=0,  # 0으로 마스킹
            width=LEG_LINE_WIDTH
        )
        
        # 오른쪽 다리
        draw.line(
            [tuple(knee_right), tuple(ankle_right)],
            fill=0,  # 0으로 마스킹
            width=LEG_LINE_WIDTH
        )
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_dress(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    드레스 카테고리에 맞는 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 드레스: 상체 + 하체 영역 (보존 부위 제외)
    clothing_categories = [
        "outer_rsleeve", "outer_lsleeve", "outer_torso", "neck",
        "inner_rsleeve", "inner_lsleeve", "inner_torso",
        "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt"
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
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
        ARM_LINE_WIDTH = int(60 / 512 * height)
        
        # 상체 영역 그리기 (몸통 부분)
        body_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            tuple(hip_right),
            tuple(hip_left)
        ]
        draw.polygon(body_poly, fill=0)  # 0으로 마스킹
        
        # 목 영역 (어깨와 코 사이)
        nose = pose_data[idx_map["nose"]]
        neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                    int((nose[1] + (shoulder_left[1] + shoulder_right[1]) / 2) / 2))
        
        neck_poly = [
            tuple(shoulder_left),
            tuple(shoulder_right),
            neck_top
        ]
        draw.polygon(neck_poly, fill=0)  # 0으로 마스킹
        
        # 팔 영역 확장 (손목까지)
        # 왼팔 소매
        draw.line(
            [tuple(shoulder_left), tuple(elbow_left), tuple(wrist_left)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
        # 오른팔 소매
        draw.line(
            [tuple(shoulder_right), tuple(elbow_right), tuple(wrist_right)],
            fill=0,  # 0으로 마스킹
            width=ARM_LINE_WIDTH,
            joint="curve"
        )
        
        # 하체 영역 확장
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        
        # 하체 영역 그리기
        lower_poly = [
            tuple(hip_left),
            tuple(hip_right),
            tuple(knee_right),
            tuple(knee_left)
        ]
        draw.polygon(lower_poly, fill=0)  # 0으로 마스킹
        
        # 무릎 아래 영역 (일반 드레스는 무릎까지)
        LEG_LINE_WIDTH = int(50 / 512 * height)
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_long_dress(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    롱 드레스 카테고리에 맞게 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
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
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
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
        draw.polygon(body_poly, fill=0)  # 0으로 마스킹
        
        # 목 영역 (어깨와 코 사이)
        nose = pose_data[idx_map["nose"]]
        neck_top = (int((shoulder_left[0] + shoulder_right[0]) / 2),
                    int((nose[1] * 0.8 + (shoulder_left[1] + shoulder_right[1]) / 2 * 0.2)))
        
        # 목 주변 더 넓게 마스킹
        neck_width = int(abs(shoulder_right[0] - shoulder_left[0]) * 0.5)
        neck_poly = [
            (shoulder_left[0] - neck_width//4, shoulder_left[1]),
            (shoulder_right[0] + neck_width//4, shoulder_right[1]),
            (neck_top[0] + neck_width//2, neck_top[1] - neck_width//3),
            (neck_top[0] - neck_width//2, neck_top[1] - neck_width//3)
        ]
        draw.polygon(neck_poly, fill=0)  # 0으로 마스킹
        
        # 팔 영역 확장 (손목까지) - 자연스럽게 넓게
        # 왼팔 소매
        sleeve_left_points = [
            (shoulder_left[0] - ARM_LINE_WIDTH//2, shoulder_left[1]),
            (elbow_left[0] - ARM_LINE_WIDTH//2, elbow_left[1]),
            (wrist_left[0] - ARM_LINE_WIDTH//3, wrist_left[1] + ARM_LINE_WIDTH//3),
            (wrist_left[0] + ARM_LINE_WIDTH//2, wrist_left[1] + ARM_LINE_WIDTH//3),
            (elbow_left[0] + ARM_LINE_WIDTH//2, elbow_left[1]),
            (shoulder_left[0] + ARM_LINE_WIDTH//2, shoulder_left[1])
        ]
        draw.polygon(sleeve_left_points, fill=0)
        
        # 오른팔 소매 - 자연스럽게 넓게
        sleeve_right_points = [
            (shoulder_right[0] - ARM_LINE_WIDTH//2, shoulder_right[1]),
            (elbow_right[0] - ARM_LINE_WIDTH//2, elbow_right[1]),
            (wrist_right[0] - ARM_LINE_WIDTH//2, wrist_right[1] + ARM_LINE_WIDTH//3),
            (wrist_right[0] + ARM_LINE_WIDTH//3, wrist_right[1] + ARM_LINE_WIDTH//3),
            (elbow_right[0] + ARM_LINE_WIDTH//2, elbow_right[1]),
            (shoulder_right[0] + ARM_LINE_WIDTH//2, shoulder_right[1])
        ]
        draw.polygon(sleeve_right_points, fill=0)
        
        # 하체 영역 확장
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        ankle_left = pose_data[idx_map["left_ankle"]]
        ankle_right = pose_data[idx_map["right_ankle"]]
        
        # 하체 영역 그리기 - 더 넓게
        lower_width = int(abs(hip_right[0] - hip_left[0]) * 1.2)
        lower_left = (hip_left[0] - lower_width//5, hip_left[1])
        lower_right = (hip_right[0] + lower_width//5, hip_right[1])
        
        # 무릎 위치도 더 넓게
        knee_left_wide = (knee_left[0] - lower_width//5, knee_left[1])
        knee_right_wide = (knee_right[0] + lower_width//5, knee_right[1])
        
        # 하체 폴리곤 (허리-무릎)
        lower_poly = [
            lower_left,
            lower_right,
            knee_right_wide,
            knee_left_wide
        ]
        draw.polygon(lower_poly, fill=0)
        
        # 롱 드레스 - 발목까지 확장
        # 발목 위치 - 더 넓게
        LEG_LINE_WIDTH = int(80 / 512 * height)
        ankle_margin = int(LEG_LINE_WIDTH * 1.5)
        
        # 스커트/드레스 하단부 (무릎-발목)
        skirt_bottom = ankle_left[1] - LEG_LINE_WIDTH//2  # 발목 위에서 끝나도록
        skirt_left = (ankle_left[0] - ankle_margin, skirt_bottom)
        skirt_right = (ankle_right[0] + ankle_margin, skirt_bottom)
        
        # 스커트 폴리곤 (무릎-발목)
        skirt_poly = [
            knee_left_wide,
            knee_right_wide,
            skirt_right,
            skirt_left
        ]
        draw.polygon(skirt_poly, fill=0)
        
        # 발목 위쪽에 자연스러운 경계선 추가
        transition_height = LEG_LINE_WIDTH
        transition_bottom = skirt_bottom + transition_height
        
        # 더 자연스러운 전환을 위한 그라데이션 효과
        steps = 10
        for i in range(steps):
            alpha = (255 * (steps - i) / steps) * 0  # 모두 0으로 설정 (마스킹)
            y_pos = skirt_bottom + i * transition_height / steps
            transition_width = ankle_margin * (steps - i) / steps
            
            left_x = ankle_left[0] - transition_width
            right_x = ankle_right[0] + transition_width
            
            draw.line([(left_x, y_pos), (right_x, y_pos)], fill=int(0), width=2)
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 디버깅: 마스크 이미지 저장 (선택 사항)
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_skirt(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    스커트 카테고리에 맞는 agnostic 이미지 생성
    """
    # 일반 스커트는 일반 하의와 동일하게 처리
    return create_agnostic_lower(
        image_path,
        parsing_json_path,
        keypoints_json_path,
        output_path
    )

def create_agnostic_short_skirt(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    짧은 스커트 카테고리에 맞는 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 짧은 스커트: 엉덩이부터 무릎 위까지
    clothing_categories = [
        "pants_hip", "skirt"
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_pil)
    
    # 키포인트 좌표 추출 (normalize 형식에서 pixel로 변환)
    pose_data = []
    for kp in keypoints:
        x, y = int(kp.get("x", 0) * width), int(kp.get("y", 0) * height)
        pose_data.append([x, y])
    
    pose_data = np.array(pose_data)
    
    # 필요한 키포인트 인덱스 매핑
    idx_map = {
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26
    }
    
    try:
        # 하체 영역 확장
        hip_right = pose_data[idx_map["right_hip"]]
        hip_left = pose_data[idx_map["left_hip"]]
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        
        # 엉덩이부터 무릎까지의 30% 정도만 사용 (짧은 스커트)
        knee_y_30_percent = int(hip_left[1] + (knee_left[1] - hip_left[1]) * 0.3)
        
        # 스커트 영역 그리기
        skirt_width = int(abs(hip_right[0] - hip_left[0]) * 1.1)  # 약간 더 넓게
        
        # 허리선에서 약간 더 넓어지도록 설정
        skirt_left = (hip_left[0] - skirt_width//10, knee_y_30_percent)
        skirt_right = (hip_right[0] + skirt_width//10, knee_y_30_percent)
        
        # 짧은 스커트 폴리곤
        skirt_poly = [
            tuple(hip_left),
            tuple(hip_right),
            skirt_right,
            skirt_left
        ]
        draw.polygon(skirt_poly, fill=0)  # 0으로 마스킹
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask

def create_agnostic_long_skirt(
    image_path,
    parsing_json_path,
    keypoints_json_path,
    output_path=None
):
    """
    긴 스커트 카테고리에 맞는 agnostic 이미지 생성
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
    
    # 4. 의류 영역(제거할 부분) 마스크 생성
    # 긴 스커트: 엉덩이부터 발목까지
    clothing_categories = [
        "pants_hip", "skirt", "pants_rsleeve", "pants_lsleeve"
    ]
    
    # 카테고리 ID 매핑
    category_ids = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13, "skirt": 14,
        "right_arm": 15, "left_arm": 16, "right_shoe": 17, "left_shoe": 18,
        "right_leg": 19, "left_leg": 20
    }
    
    # 빈 마스크 생성 (0: 마스킹, 255: 유지)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 파싱 결과에서 의류 영역 마스킹
    class_masks = parsing_data.get("class_masks", {})
    for class_name, coords in class_masks.items():
        if class_name in clothing_categories and coords:
            for x, y in coords:
                if 0 <= x < width and 0 <= y < height:
                    mask[y, x] = 0
    
    # 5. 키포인트를 이용해 의류 영역 마스크 확장
    # PIL 이미지로 변환하여 드로잉 작업
    mask_pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_pil)
    
    # 키포인트 좌표 추출 (normalize 형식에서 pixel로 변환)
    pose_data = []
    for kp in keypoints:
        x, y = int(kp.get("x", 0) * width), int(kp.get("y", 0) * height)
        pose_data.append([x, y])
    
    pose_data = np.array(pose_data)
    
    # 필요한 키포인트 인덱스 매핑
    idx_map = {
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28
    }
    
    try:
        # 하체 영역 확장
        hip_right = pose_data[idx_map["right_hip"]]
        hip_left = pose_data[idx_map["left_hip"]]
        knee_left = pose_data[idx_map["left_knee"]]
        knee_right = pose_data[idx_map["right_knee"]]
        ankle_left = pose_data[idx_map["left_ankle"]]
        ankle_right = pose_data[idx_map["right_ankle"]]
        
        # 롱 스커트 영역 그리기
        skirt_width = int(abs(hip_right[0] - hip_left[0]) * 1.3)
        
        # 허리선에서 발목까지 약간 넓어지도록 설정
        ankle_mid_x = (ankle_left[0] + ankle_right[0]) // 2
        ankle_mid_y = (ankle_left[1] + ankle_right[1]) // 2
        
        # 발목 위치 설정 - 발목 위에서 약간 여유를 두고 끝남
        ankle_margin = int(skirt_width * 0.2)
        skirt_bottom = ankle_mid_y - int(height * 0.05)
        skirt_left = (ankle_mid_x - skirt_width//2, skirt_bottom)
        skirt_right = (ankle_mid_x + skirt_width//2, skirt_bottom)
        
        # 스커트 폴리곤 (사다리꼴 형태)
        skirt_poly = [
            tuple(hip_left),
            tuple(hip_right),
            skirt_right,
            skirt_left
        ]
        draw.polygon(skirt_poly, fill=0)  # 0으로 마스킹
        
    except (IndexError, KeyError) as e:
        print(f"키포인트 처리 중 오류 발생: {str(e)}")
    
    # PIL 이미지를 numpy 배열로 변환
    mask_enhanced = np.array(mask_pil)
    
    # 마스크 개선 (노이즈 제거 및 경계 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)
    mask_enhanced = cv2.dilate(mask_enhanced, kernel, iterations=2)  # 마스크 확장
    mask_enhanced = cv2.GaussianBlur(mask_enhanced, (11, 11), 0)
    mask_enhanced = cv2.threshold(mask_enhanced, 127, 255, cv2.THRESH_BINARY)[1]
    
    # 6. Agnostic 이미지 생성 (의류 영역을 0으로 마스킹)
    # 마스크 영역의 RGB 값을 0으로 설정
    agnostic_image = image_rgb.copy()
    
    # 마스크가 0인 영역(의류 영역)을 이미지에서 0으로 설정
    agnostic_image[mask_enhanced == 0] = [0, 0, 0]
    
    # PIL 이미지로 변환
    agnostic_pil = Image.fromarray(agnostic_image)
    
    # 7. IDM VTON용 마스크 생성 (의류:1, 배경:0)
    idm_mask = (mask_enhanced == 0).astype(np.uint8)
    
    # 결과 저장
    if output_path:
        mask_path = output_path.replace('.jpg', '_mask.png')
        cv2.imwrite(mask_path, idm_mask * 255)
        agnostic_pil.save(output_path)
        print(f"마스크 이미지 저장: {mask_path}")
        print(f"Agnostic 이미지 저장: {output_path}")
    
    return agnostic_pil, idm_mask