"""
DensePose API 클라이언트
별도 가상환경에서 실행 중인 DensePose API 서버에 요청하는 클라이언트
"""

import os
import requests
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Union


class DensePoseAPIClient:
    """DensePose API 클라이언트 (별도 가상환경의 모델 활용)"""

    def __init__(self, api_url="http://27.117.208.40:29246"):
        """
        Parameters:
        -----------
        api_url : str
            DensePose API 서버 주소 (기본값: http://localhost:8001)
        """
        self.api_url = api_url
        self.process_endpoint = f"{api_url}/process"
        self.process_file_endpoint = f"{api_url}/process/file"

    def _check_connection(self):
        """API 서버 연결 확인"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("densepose_loaded", False):
                    return True
                else:
                    print("경고: API 서버에 DensePose 모델이 로드되지 않았습니다.")
                    return False
            else:
                print(f"오류: API 서버 응답 코드 {response.status_code}")
                return False
        except Exception as e:
            print(f"API 서버 연결 오류: {e}")
            return False

    def _encode_image(self, image):
        """이미지를 base64로 인코딩"""
        if isinstance(image, np.ndarray):
            # OpenCV BGR -> RGB 변환
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # numpy -> PIL
            image = Image.fromarray(image)

        # PIL -> base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str

    def process_image(self, image, options=None):
        """
        이미지에 대한 DensePose 분석 실행

        Parameters:
        -----------
        image : numpy.ndarray or PIL.Image
            분석할 이미지
        options : dict, optional
            API 요청 옵션

        Returns:
        --------
        dict
            DensePose 분석 결과 (Base64 이미지 데이터 포함)
        """
        # 기본 옵션 설정
        if options is None:
            options = {
                "include_body_parts": True,
                "include_uv_coordinates": True,
                "sample_uv": None,  # 모든 UV 좌표 가져오기
                "create_visualization": True,
                "output_directory": "output"
            }

        # 연결 확인
        try:
            # 이미지 인코딩
            img_base64 = self._encode_image(image)

            # API 요청
            payload = {
                "image_base64": img_base64,
                "options": options
            }

            response = requests.post(self.process_endpoint, json=payload)

            if response.status_code != 200:
                return {
                    "status": "error",
                    "densepose_data": {
                        "error": f"API 요청 실패: {response.status_code} - {response.text}"
                    }
                }

            result = response.json()

            # Windows 경로를 Linux에서 바로 사용할 수 없으므로 validation_path를 처리
            if "densepose_data" in result and "visualization_path" in result["densepose_data"]:
                path = result["densepose_data"]["visualization_path"]
                # Windows 경로를 Linux 경로로 변환하지 않고, 대신 Base64 데이터 사용

                # DensePose API에서 Base64 이미지 데이터가 없는 경우
                if "visualization_base64" not in result and "visualization_base64" not in result.get("densepose_data", {}):
                    # 경로 대신 Base64 데이터 포함 표시
                    result["densepose_data"]["windows_path"] = path
                    result["densepose_data"]["visualization_path"] = None
                    print("경고: DensePose API가 Base64 이미지 데이터를 제공하지 않았습니다. 경로 기반 접근이 실패할 수 있습니다.")

            # 모든 경우에 Base64 데이터가 응답의 최상위 레벨에 있으면 densepose_data 내부로 이동
            if "visualization_base64" in result and "densepose_data" in result:
                result["densepose_data"]["visualization_base64"] = result["visualization_base64"]

            return result
        except Exception as e:
            return {
                "status": "error",
                "densepose_data": {
                    "error": f"DensePose API 처리 오류: {str(e)}"
                }
            }

    def process_file(self, image_path, **kwargs):
        """
        이미지 파일에 대한 DensePose 분석 실행

        Parameters:
        -----------
        image_path : str
            분석할 이미지 파일 경로
        **kwargs :
            API 요청 파라미터

        Returns:
        --------
        dict
            DensePose 분석 결과
        """
        # 연결 확인
        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post(self.process_file_endpoint, files=files, data=kwargs)

            if response.status_code != 200:
                return {
                    "status": "error",
                    "densepose_data": {
                        "error": f"API 요청 실패: {response.status_code} - {response.text}"
                    }
                }

            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "densepose_data": {
                    "error": f"DensePose API 처리 오류: {str(e)}"
                }
            }

    def scale_uv_coordinates(self, densepose_data, original_size):
        """
        DensePose UV 좌표를 원본 이미지 크기로 변환 (Base64 데이터 유지)

        Parameters:
        -----------
        densepose_data : dict
            API로부터 받은 DensePose 결과 데이터
        original_size : tuple
            원본 이미지 크기 (width, height)

        Returns:
        --------
        dict
            원본 크기에 맞게 조정된 DensePose 데이터
        """
        if not densepose_data or "densepose_data" not in densepose_data:
            return densepose_data

        data = densepose_data["densepose_data"]

        # Base64 이미지 데이터 백업 (스케일링 후에도 보존)
        visualization_base64 = data.get("visualization_base64")

        # 원본 크기 정보
        orig_width, orig_height = original_size
        api_width = data.get("image_width", orig_width)
        api_height = data.get("image_height", orig_height)

        # 크기가 이미 동일하면 변환 필요 없음
        if api_width == orig_width and api_height == orig_height:
            return densepose_data

        # 스케일 계수 계산
        scale_x = orig_width / api_width
        scale_y = orig_height / api_height

        # 바운딩 박스 스케일링
        if "bbox" in data:
            x0, y0, x1, y1 = data["bbox"]
            data["bbox"] = [
                int(x0 * scale_x),
                int(y0 * scale_y),
                int(x1 * scale_x),
                int(y1 * scale_y)
            ]

        # UV 좌표 스케일링
        if "uv_coordinates" in data:
            for uv_point in data["uv_coordinates"]:
                uv_point["x"] = int(uv_point["x"] * scale_x)
                uv_point["y"] = int(uv_point["y"] * scale_y)

        # 이미지 크기 업데이트
        data["image_width"] = orig_width
        data["image_height"] = orig_height

        # Base64 데이터 복원
        if visualization_base64:
            data["visualization_base64"] = visualization_base64

        return densepose_data
    def create_model_parse_from_densepose(self, densepose_data, size=(384, 512)):
        """
        DensePose 결과로부터 model_parse 이미지 생성

        Parameters:
        -----------
        densepose_data : dict
            API로부터 받은 DensePose 결과 데이터
        size : tuple
            생성할 이미지 크기 (width, height)

        Returns:
        --------
        PIL.Image
            인체 파싱 이미지
        """
        if "densepose_data" not in densepose_data:
            return None

        data = densepose_data["densepose_data"]

        # UV 좌표가 없으면 처리 불가
        if "uv_coordinates" not in data:
            return None

        # 이미지 생성
        width, height = size
        model_parse = Image.new('L', (width, height), 0)

        # UV 좌표로부터 파싱 이미지 생성
        for uv_point in data["uv_coordinates"]:
            x, y = uv_point["x"], uv_point["y"]
            part_id = uv_point["part_id"]

            # 이미지 크기 내에 있는지 확인
            if 0 <= x < width and 0 <= y < height:
                model_parse.putpixel((x, y), part_id)

        return model_parse