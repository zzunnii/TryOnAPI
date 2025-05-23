import pandas as pd
import numpy as np
import cv2
import os

# ✅ 경로 설정
parquet_path = r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\preprocessed\model\train\model_person_000000\model_info.parquet"
output_dir = os.path.dirname(parquet_path)
W, H = 720, 1280  # 너비(Width) 720, 높이(Height) 1280

# ✅ parquet 불러오기
df = pd.read_parquet(parquet_path)
U = np.zeros((H, W), dtype=np.float32)
V = np.zeros((H, W), dtype=np.float32)
I = np.zeros((H, W), dtype=np.uint8)
depth = np.zeros((H, W), dtype=np.float32)

# 좌표 정보를 사용하여 복원
for i, row in df.iterrows():
    y, x = int(row['row']), int(row['col'])
    # 범위 체크 추가
    if 0 <= y < H and 0 <= x < W:
        U[y, x] = row['U']
        V[y, x] = row['V']
        I[y, x] = row['I']
        depth[y, x] = row['depth']
    else:
        print(f"범위 초과 인덱스 무시: y={y}, x={x}, 범위는 (0-{H-1}, 0-{W-1})")

# ✅ 시각화 함수
def save_visual_map(img, save_path, is_float=True, log=False):
    if np.isnan(img).all() or np.isinf(img).all():
        print(f"❌ 시각화 불가 (NaN/Inf): {save_path}")
        return
    img = np.clip(img, 0, None)
    if log:
        img = np.log1p(img)
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    cv2.imwrite(save_path, norm.astype(np.uint8) if is_float else norm)

# ✅ 저장
cv2.imwrite(os.path.join(output_dir, "model_U.png"), (U * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "model_V.png"), (V * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, "model_I.png"), (I * (255. / I.max())).astype(np.uint8))
save_visual_map(depth, os.path.join(output_dir, "model_depth.png"))
save_visual_map(depth, os.path.join(output_dir, "model_depth_log.png"), log=True)

print("✅ 시각화 완료")
