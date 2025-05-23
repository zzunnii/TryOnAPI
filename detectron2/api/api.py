"""
DensePose → IDM‑VTON API
-----------------------
Detectron2 DensePose 로부터 UV 좌표를 추출하여 IDM‑VTON 파이프라인이
필요로 하는 JSON 형식으로 반환한다.  (depth 제거)
추가로 디버깅을 위한 컬러맵 시각화 이미지를 생성한다.
"""

# ---------------------------------------------------------------------------
# Path setup ----------------------------------------------------------------
# ---------------------------------------------------------------------------
import sys
import os
from typing import Dict, Any, List, Optional, TYPE_CHECKING

# DensePose project 경로를 PYTHONPATH 에 포함
sys.path.append(r"C:\Users\tjdwn\detectron2\projects\DensePose")

# detectron2 루트 경로(현재 파일 위치 기준 두 단계 위) 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ---------------------------------------------------------------------------
# Imports -------------------------------------------------------------------
# ---------------------------------------------------------------------------
import base64
import json
import tempfile
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Detectron2 + DensePose
try:
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.structures.instances import Instances

    from densepose.config import add_densepose_config
    from densepose.converters import ToChartResultConverter

    DETECTRON2_AVAILABLE = True
except ImportError as e:
    print(f"Detectron2 또는 DensePose 를 사용할 수 없습니다: {e}")
    DETECTRON2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants & configuration --------------------------------------------------
# ---------------------------------------------------------------------------
DENSEPOSE_CFG_PATH = (
    "C:/Users/tjdwn/detectron2/projects/DensePose/configs/"  # 수정 가능
    "densepose_rcnn_R_50_FPN_s1x.yaml"
)
DENSEPOSE_MODEL_PATH = (
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/"  # noqa
    "165712039/model_final_162be9.pkl"
)

# ---------------------------------------------------------------------------
# FastAPI --------------------------------------------------------------------
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DensePose → IDM‑VTON API",
    description="DensePose UV 추출 및 컬러맵 시각화 API (depth 없음)",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response models --------------------------------------------------
# ---------------------------------------------------------------------------
class DensePoseRequest(BaseModel):
    image_base64: str  # 반드시 포함
    options: Dict[str, Any] = {
        "include_body_parts": True,        # part‑id (I) 기반 컬러맵
        "include_uv_coordinates": True,    # UV 기반 컬러맵
        "sample_uv": None,               # uv_coordinates 샘플 수 (None = 전체)
        "create_visualization": True,      # 디버깅 이미지 저장
        "output_directory": "output"        # 저장 디렉토리
    }


class DensePoseResponse(BaseModel):
    status: str
    densepose_data: Optional[Dict[str, Any]] = None
    visualization_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Globals -------------------------------------------------------------------
# ---------------------------------------------------------------------------
densepose_predictor: Optional[DefaultPredictor] = None
converter: Optional[ToChartResultConverter] = None


# ---------------------------------------------------------------------------
# Utils ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
def initialize_models() -> bool:
    """Load DensePose model lazily."""
    global densepose_predictor, converter

    if not DETECTRON2_AVAILABLE:
        return False

    if densepose_predictor is not None:
        return True  # already loaded

    try:
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(DENSEPOSE_CFG_PATH)
        cfg.MODEL.WEIGHTS = DENSEPOSE_MODEL_PATH
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        densepose_predictor = DefaultPredictor(cfg)
        converter = ToChartResultConverter()

        print(f"DensePose 모델 로드 완료. Device = {cfg.MODEL.DEVICE}")
        return True
    except Exception as e:
        print(f"DensePose 초기화 오류: {e}")
        return False


def decode_base64_image(data: str) -> np.ndarray:
    """base64 → numpy(BGR)"""
    try:
        img = np.array(Image.open(BytesIO(base64.b64decode(data))))
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        raise ValueError(f"Base64 디코딩 실패: {e}")


# ---------------------------------------------------------------------------
# Core processing -----------------------------------------------------------
# ---------------------------------------------------------------------------

def run_densepose(
    image_bgr: np.ndarray,
    opts: Dict[str, Any],
) -> Dict[str, Any]:
    """DensePose inference + visualization + UV list build."""
    if not initialize_models():
        return {"error": "DensePose 모델을 불러올 수 없습니다."}

    H, W = image_bgr.shape[:2]

    outputs = densepose_predictor(image_bgr)
    instances: Instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return {
            "error": "사람을 찾지 못했습니다.",
            "image_width": W,
            "image_height": H,
            "num_persons": 0,
        }

    # ----- 첫 번째 사람만 사용 (single‑person use‑case) --------------------
    dp = instances.pred_densepose[0]
    bbox = instances.pred_boxes[0]  # xyxy
    dp_result = converter.convert(dp, bbox)

    # 풀사이즈 I, U, V 맵 만들기
    I = np.zeros((H, W), dtype=np.uint8)
    U = np.zeros((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)

    x0, y0, x1, y1 = map(int, bbox.tensor[0])
    h_iuv, w_iuv = dp_result.labels.shape
    I[y0:y0 + h_iuv, x0:x0 + w_iuv] = dp_result.labels
    U[y0:y0 + h_iuv, x0:x0 + w_iuv] = dp_result.uv[0]
    V[y0:y0 + h_iuv, x0:x0 + w_iuv] = dp_result.uv[1]

    # ----- uv_coordinates list -------------------------------------------
    mask = I > 0
    rows, cols = np.where(mask)
    sample_uv = int(opts.get("sample_uv", 0) or 0)
    if sample_uv and len(rows) > sample_uv:
        indices = np.random.choice(len(rows), sample_uv, replace=False)
        rows, cols = rows[indices], cols[indices]

    uv_coordinates: List[Dict[str, Any]] = []
    for r, c in zip(rows, cols):
        uv_coordinates.append({
            "x": int(c),
            "y": int(r),
            "u": float(U[r, c]),
            "v": float(V[r, c]),
            "part_id": int(I[r, c]),
        })

    # ----- visualization ---------------------------------------------------
    vis_path = None
    vis_base64 = None  # 추가: Base64 데이터 변수

    if opts.get("create_visualization", True):
        out_dir = opts.get("output_directory", "output")
        os.makedirs(out_dir, exist_ok=True)

        # 기존 시각화 코드 그대로 유지
        out_dir = os.path.abspath(out_dir)
        vis_path = os.path.join(out_dir, f"dp_segm_{int(time.time())}.png")

        # IDM VTON gradio 데모와 동일한 UV 기반 RGB 매핑
        debug_img = np.zeros((H, W, 3), dtype=np.uint8)
        mask = I > 0
        I_mapped = np.zeros_like(I)
        I_mapped[mask] = I[mask]
        color_mapped = cv2.applyColorMap(
            ((I_mapped / 24.0) * 255).astype(np.uint8),
            cv2.COLORMAP_PARULA
        )
        debug_img[mask] = color_mapped[mask]

        # 이미지 저장
        cv2.imwrite(vis_path, debug_img)

        # 추가: 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.png', debug_img)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')

        # 파일 존재 확인
        if not os.path.exists(vis_path):
            print(f"경고: 시각화 파일이 생성되지 않았습니다: {vis_path}")

    # Base64 데이터 추가
    return {
        "image_width": W,
        "image_height": H,
        "num_persons": len(instances),
        "bbox": [x0, y0, x1, y1],
        "uv_coordinates": uv_coordinates,
        "visualization_path": vis_path,
        "visualization_base64": vis_base64  # 추가: Base64 데이터
    }

# ---------------------------------------------------------------------------
# API End‑points ------------------------------------------------------------
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    initialize_models()


@app.get("/")
async def root():
    return {"message": "DensePose → IDM‑VTON API"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "densepose_loaded": densepose_predictor is not None,
    }


@app.post("/process", response_model=DensePoseResponse)
async def process_base64(req: DensePoseRequest = Body(...)):
    try:
        img_bgr = decode_base64_image(req.image_base64)
        result = run_densepose(img_bgr, req.options)

        if "error" in result:
            raise RuntimeError(result["error"])

        resp = {
            "status": "success",
            "densepose_data": result,
            "visualization_base64": result.get("visualization_base64"),  # Base64 추가
        }
        if result.get("visualization_path"):
            resp["visualization_url"] = result["visualization_path"]
        return resp
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
@app.post("/process/file", response_model=DensePoseResponse)
async def process_file(
        file: UploadFile = File(...),
        include_body_parts: bool = True,
        include_uv_coordinates: bool = True,
        sample_uv: int = 10000,
        create_visualization: bool = True,
        output_directory: str = "output",
):
    os.makedirs(output_directory, exist_ok=True)
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise HTTPException(400, "이미지 로드 실패")
        os.unlink(tmp_path)

        # gradio 데모와 유사한 'dp_segm' 형식 시각화 생성
        H, W = img_bgr.shape[:2]
        outputs = densepose_predictor(img_bgr)
        instances: Instances = outputs["instances"].to("cpu")

        # 시각화 생성 (dp_segm 형식)
        debug_img = np.zeros((H, W, 3), dtype=np.uint8)
        if len(instances) > 0:
            dp = instances.pred_densepose[0]
            bbox = instances.pred_boxes[0]
            dp_result = converter.convert(dp, bbox)

            # 전체 이미지 크기 IUV 맵 생성
            x0, y0, x1, y1 = map(int, bbox.tensor[0])
            h_iuv, w_iuv = dp_result.labels.shape
            I = np.zeros((H, W), dtype=np.uint8)
            I[y0:y0 + h_iuv, x0:x0 + w_iuv] = dp_result.labels

            # 마스크 및 컬러맵 생성 (gradio 데모의 dp_segm과 유사하게)
            mask = I > 0
            I_mapped = np.zeros_like(I)
            I_mapped[mask] = I[mask]
            color_mapped = cv2.applyColorMap(
                ((I_mapped / 24.0) * 255).astype(np.uint8),
                cv2.COLORMAP_PARULA
            )
            debug_img[mask] = color_mapped[mask]

        # 원본과 블렌드

        # 원본 API 응답에 pose_img를 직접 추가 (base64 형식으로)
        vis_path = os.path.join(output_directory, f"dp_segm_{int(time.time())}.jpg")
        cv2.imwrite(vis_path, debug_img)

        # 기존 UV 좌표 처리
        opts = {
            "include_body_parts": include_body_parts,
            "include_uv_coordinates": include_uv_coordinates,
            "sample_uv": sample_uv,
            "create_visualization": False,  # 이미 위에서 생성했으므로 중복 생성 방지
            "output_directory": output_directory,
        }
        result = run_densepose(img_bgr, opts)
        result["visualization_path"] = vis_path  # 새로운 키 추가

        resp = {
            "status": "success",
            "densepose_data": result,
            "visualization_url": vis_path,
        }
        return resp
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
